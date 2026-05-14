package llm

// GovernorBackend talks to a deployed Pennsieve LLM Governor over its HTTPS
// Function URL. This replaces the previous LambdaBackend which used direct
// `lambda:Invoke` — the governor migrated to `lambdaurl.Start` with response
// streaming, so the only supported invocation path is now HTTPS + SigV4.
//
// The protocol surface is the Anthropic Messages API (POST /v1/messages),
// plus governor-specific embed/rerank endpoints. SigV4 signs the request
// against the "lambda" service (Function URLs are signed as Lambda
// invocations, not API Gateway).

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
)

// signingService is "lambda" because Function URLs are signed as Lambda
// invocations (not "execute-api" — that's API Gateway).
const signingService = "lambda"

// defaultRegion is used when neither AWS_REGION nor an explicit option is set.
// Governors are typically deployed in us-east-1; override via
// WithGovernorRegion if your compute node is elsewhere.
const defaultRegion = "us-east-1"

// GovernorBackend is the canonical backend for production use against a
// deployed governor. It supersedes LambdaBackend (now removed).
type GovernorBackend struct {
	url        string
	region     string
	httpClient *http.Client
	creds      aws.CredentialsProvider
}

// GovernorBackendOption configures a GovernorBackend.
type GovernorBackendOption func(*GovernorBackend)

// WithGovernorURL overrides the governor URL.
// Default: $LLM_GOVERNOR_URL.
func WithGovernorURL(u string) GovernorBackendOption {
	return func(b *GovernorBackend) { b.url = u }
}

// WithGovernorRegion overrides the SigV4 signing region.
// Default: $AWS_REGION, falling back to us-east-1.
func WithGovernorRegion(r string) GovernorBackendOption {
	return func(b *GovernorBackend) { b.region = r }
}

// WithGovernorHTTPClient overrides the http.Client used to talk to the
// governor. Useful for testing or for custom timeout/transport settings.
func WithGovernorHTTPClient(c *http.Client) GovernorBackendOption {
	return func(b *GovernorBackend) { b.httpClient = c }
}

// WithGovernorCredentials overrides the AWS credentials provider used for
// SigV4. Default: the default credential chain (env vars, IAM role, etc.).
func WithGovernorCredentials(c aws.CredentialsProvider) GovernorBackendOption {
	return func(b *GovernorBackend) { b.creds = c }
}

// NewGovernorBackend constructs a GovernorBackend. Reads $LLM_GOVERNOR_URL
// and $AWS_REGION from the environment by default; both can be overridden
// via options. AWS credentials are loaded from the default chain.
//
// Returns an error if no URL is configured or if AWS config cannot be loaded.
func NewGovernorBackend(ctx context.Context, opts ...GovernorBackendOption) (*GovernorBackend, error) {
	b := &GovernorBackend{
		url:    os.Getenv("LLM_GOVERNOR_URL"),
		region: os.Getenv("AWS_REGION"),
	}
	for _, opt := range opts {
		opt(b)
	}
	if b.region == "" {
		b.region = defaultRegion
	}
	if b.url == "" {
		return nil, fmt.Errorf("governor URL is required (set LLM_GOVERNOR_URL or use WithGovernorURL)")
	}
	// Strip trailing slash so URL joins are consistent.
	for len(b.url) > 0 && b.url[len(b.url)-1] == '/' {
		b.url = b.url[:len(b.url)-1]
	}
	if b.creds == nil {
		cfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to load AWS config for SigV4: %w", err)
		}
		b.creds = cfg.Credentials
		if b.region == defaultRegion && cfg.Region != "" {
			b.region = cfg.Region
		}
	}
	if b.httpClient == nil {
		// Long timeout — Opus calls with extended thinking + large prompts
		// can take >2 minutes. Function URL streaming has its own per-chunk
		// behavior; this just bounds the whole-response wait.
		b.httpClient = &http.Client{Timeout: 15 * time.Minute}
	}
	return b, nil
}

// Invoke sends a non-streaming chat request via POST /v1/messages.
func (b *GovernorBackend) Invoke(ctx context.Context, req *InvokeRequest) (*InvokeResponse, error) {
	anthReq := buildAnthropicRequestFromInvoke(req)
	body, err := json.Marshal(anthReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Anthropic request: %w", err)
	}

	respBody, err := b.do(ctx, http.MethodPost, "/v1/messages", body)
	if err != nil {
		return nil, err
	}

	var anthResp anthropicResponse
	if err := json.Unmarshal(respBody, &anthResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Anthropic response: %w", err)
	}
	return buildInvokeResponseFromAnthropic(&anthResp), nil
}

// CheckBudget queries GET /v1/budget?execution_run_id=X.
func (b *GovernorBackend) CheckBudget(ctx context.Context, executionRunID string) (*CheckBudgetResponse, error) {
	q := url.Values{}
	if executionRunID != "" {
		q.Set("execution_run_id", executionRunID)
	}
	path := "/v1/budget"
	if encoded := q.Encode(); encoded != "" {
		path = path + "?" + encoded
	}
	respBody, err := b.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	var resp CheckBudgetResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal CheckBudget response: %w", err)
	}
	return &resp, nil
}

// ListModels queries GET /v1/models.
func (b *GovernorBackend) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	respBody, err := b.do(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return nil, err
	}
	var resp ListModelsResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ListModels response: %w", err)
	}
	return &resp, nil
}

// do builds, SigV4-signs, and sends an HTTP request to the governor. Returns
// the response body on success. Maps governor error envelopes to *GovernorError.
func (b *GovernorBackend) do(ctx context.Context, method, path string, body []byte) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		reqBody = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, b.url+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	if err := b.sign(ctx, req, body); err != nil {
		return nil, fmt.Errorf("SigV4 sign failed: %w", err)
	}

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode >= 400 {
		// Try to parse as a governor error envelope so callers can pattern
		// match on Code (e.g. "model_not_allowed", "budget_exceeded").
		var errResp ErrorResponse
		if jsonErr := json.Unmarshal(respBody, &errResp); jsonErr == nil && errResp.Error != "" {
			return nil, &GovernorError{
				Code:            errResp.Error,
				Msg:             errResp.Message,
				AllowedModels:   errResp.AllowedModels,
				BudgetRemaining: errResp.BudgetRemaining,
				RetryAfterSec:   errResp.RetryAfterSec,
			}
		}
		return nil, fmt.Errorf("governor returned HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// sign attaches a SigV4 Authorization header to req. payload is the raw
// request body bytes used to compute the payload hash; pass nil for GETs.
func (b *GovernorBackend) sign(ctx context.Context, req *http.Request, payload []byte) error {
	creds, err := b.creds.Retrieve(ctx)
	if err != nil {
		return fmt.Errorf("failed to retrieve AWS credentials: %w", err)
	}

	hash := sha256.Sum256(payload) // sha256 of empty input is the well-known empty-payload hash
	payloadHash := hex.EncodeToString(hash[:])

	signer := v4.NewSigner()
	return signer.SignHTTP(ctx, creds, req, payloadHash, signingService, b.region, time.Now())
}
