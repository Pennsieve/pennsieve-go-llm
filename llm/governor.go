// Package llm is a thin configuration helper for the Pennsieve LLM Governor.
//
// The package returns a pre-configured *anthropic.Client (from
// github.com/anthropics/anthropic-sdk-go) whose transport dispatches every
// request to the governor via lambda:InvokeWithResponseStream (wrapping each
// request in a LambdaFunctionURLRequest envelope), with the x-execution-run-id
// header wired up. There is no public Function URL — all governor traffic
// stays on private AWS APIs, so this works in isolated/no-internet (compliant)
// VPCs too. Users interact with the official Anthropic SDK directly —
// streaming, tool use, prompt caching, extended thinking, every future
// Anthropic feature works without any wrapping on our side.
//
// The governor is reached by Lambda function name ($LLM_GOVERNOR_FUNCTION_NAME),
// which the Pennsieve platform injects into every LLM-enabled processor. This
// is the canonical, provider-neutral, all-deployment-modes path; off-the-shelf
// HTTP tools (Claude Code, langchain) instead use the in-task sidecar via
// $ANTHROPIC_BASE_URL.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
)

const defaultRegion = "us-east-1"

// lambdaInvokeBaseURL is the placeholder base URL handed to the Anthropic SDK
// when the governor is reached via lambda:Invoke. The lambda transport ignores
// the host entirely (it dispatches by function name) and uses only the request
// path/query, so the value just has to be a syntactically valid URL.
const lambdaInvokeBaseURL = "http://governor.lambda.invoke"

// Governor is a configuration helper for talking to the Pennsieve LLM Governor.
type Governor struct {
	functionName   string
	url            string
	executionRunID string
	region         string
	creds          aws.CredentialsProvider
	httpClient     *http.Client
	client         anthropic.Client
}

// Option configures a Governor at construction time.
type Option func(*Governor)

// WithFunctionName overrides the governor Lambda function name reached via
// lambda:InvokeWithResponseStream. Default: $LLM_GOVERNOR_FUNCTION_NAME.
func WithFunctionName(name string) Option {
	return func(g *Governor) { g.functionName = name }
}

// WithBaseURL overrides the Anthropic SDK base URL. Only meaningful together
// with WithHTTPClient (e.g. pointing the SDK at an httptest server in unit
// tests); in production the lambda transport ignores the host.
func WithBaseURL(u string) Option {
	return func(g *Governor) { g.url = u }
}

// WithExecutionRunID sets the execution-run-id attached as the
// x-execution-run-id header on every request. Default: $EXECUTION_RUN_ID.
func WithExecutionRunID(id string) Option {
	return func(g *Governor) { g.executionRunID = id }
}

// WithRegion overrides the SigV4 signing region. Default: $AWS_REGION or
// us-east-1.
func WithRegion(r string) Option {
	return func(g *Governor) { g.region = r }
}

// WithCredentials overrides the AWS credentials provider used for SigV4.
// Default: the default credential chain (env vars, IAM role, etc.).
func WithCredentials(c aws.CredentialsProvider) Option {
	return func(g *Governor) { g.creds = c }
}

// WithHTTPClient overrides the http.Client used for the underlying
// anthropic.Client. Useful for testing (e.g. against httptest.NewServer)
// or for custom timeouts/transports. When provided, SigV4 signing is
// NOT applied — you bring your own auth.
func WithHTTPClient(c *http.Client) Option {
	return func(g *Governor) { g.httpClient = c }
}

// New constructs a Governor. The governor is reached by Lambda function name
// ($LLM_GOVERNOR_FUNCTION_NAME or WithFunctionName) via
// lambda:InvokeWithResponseStream; without one, returns an error. For tests
// that don't need a real governor, use WithHTTPClient (+ WithBaseURL) to point
// the Anthropic SDK at an httptest.NewServer.
func New(ctx context.Context, opts ...Option) (*Governor, error) {
	g := &Governor{
		functionName:   os.Getenv("LLM_GOVERNOR_FUNCTION_NAME"),
		executionRunID: os.Getenv("EXECUTION_RUN_ID"),
		region:         os.Getenv("AWS_REGION"),
	}
	for _, opt := range opts {
		opt(g)
	}
	if g.region == "" {
		g.region = defaultRegion
	}

	switch {
	case g.httpClient != nil:
		// Explicit transport injection (tests / advanced use). Use the
		// provided client + base URL as-is; no lambda wiring.
		if g.url == "" {
			g.url = lambdaInvokeBaseURL
		}
	case g.functionName != "":
		// Canonical path: dispatch through lambda:InvokeWithResponseStream.
		lam, err := g.newLambdaClient(ctx)
		if err != nil {
			return nil, err
		}
		g.httpClient = &http.Client{
			Transport: &lambdaInvokeTransport{client: lam, functionName: g.functionName},
			Timeout:   15 * time.Minute,
		}
		g.url = lambdaInvokeBaseURL
	default:
		return nil, fmt.Errorf("no governor configured: set $LLM_GOVERNOR_FUNCTION_NAME (or use WithFunctionName / WithHTTPClient)")
	}
	g.url = strings.TrimRight(g.url, "/")

	// Build the underlying anthropic.Client. We pass a placeholder API key
	// because the SDK requires one; the real auth is the IAM identity of the
	// lambda:Invoke call (or whatever the injected http.Client provides).
	clientOpts := []option.RequestOption{
		option.WithBaseURL(g.url),
		option.WithAPIKey("placeholder-governor-handles-auth"),
		option.WithHTTPClient(g.httpClient),
	}
	if g.executionRunID != "" {
		clientOpts = append(clientOpts, option.WithHeaderAdd("x-execution-run-id", g.executionRunID))
	}
	g.client = anthropic.NewClient(clientOpts...)

	return g, nil
}

// newLambdaClient builds the AWS Lambda client used by the invoke transport,
// honoring an explicit region and credentials provider when set.
func (g *Governor) newLambdaClient(ctx context.Context) (*lambda.Client, error) {
	cfgOpts := []func(*config.LoadOptions) error{config.WithRegion(g.region)}
	if g.creds != nil {
		cfgOpts = append(cfgOpts, config.WithCredentialsProvider(g.creds))
	}
	cfg, err := config.LoadDefaultConfig(ctx, cfgOpts...)
	if err != nil {
		return nil, fmt.Errorf("load AWS config: %w", err)
	}
	return lambda.NewFromConfig(cfg), nil
}

// Client returns the underlying *anthropic.Client. Use this for all chat
// operations:
//
//	resp, err := gov.Client().Messages.New(ctx, anthropic.MessageNewParams{...})
//
// Streaming, tool use, prompt caching, etc. all work — see the
// anthropic-sdk-go documentation.
func (g *Governor) Client() *anthropic.Client {
	return &g.client
}

// URL returns the Anthropic SDK base URL. For the lambda-invoke transport this
// is a placeholder; the transport dispatches by function name.
func (g *Governor) URL() string { return g.url }

// FunctionName returns the governor Lambda function name reached via
// lambda:Invoke (empty when an explicit HTTP client was injected).
func (g *Governor) FunctionName() string { return g.functionName }

// ExecutionRunID returns the execution run ID attached to every request.
func (g *Governor) ExecutionRunID() string { return g.executionRunID }

// CheckBudget queries GET /v1/budget?execution_run_id=X.
func (g *Governor) CheckBudget(ctx context.Context) (*CheckBudgetResponse, error) {
	q := url.Values{}
	if g.executionRunID != "" {
		q.Set("execution_run_id", g.executionRunID)
	}
	path := "/v1/budget"
	if encoded := q.Encode(); encoded != "" {
		path = path + "?" + encoded
	}
	body, err := g.rawGet(ctx, path)
	if err != nil {
		return nil, err
	}
	var resp CheckBudgetResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal CheckBudget response: %w", err)
	}
	return &resp, nil
}

// ListModels queries GET /v1/models.
func (g *Governor) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	body, err := g.rawGet(ctx, "/v1/models")
	if err != nil {
		return nil, err
	}
	var resp ListModelsResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal ListModels response: %w", err)
	}
	return &resp, nil
}

// EFSDocument returns a content block referencing a file on EFS. The
// governor reads the file server-side with execution-scoped access controls,
// so the caller doesn't need to base64-encode it into the request payload.
//
// Note: returns map[string]any (raw JSON shape) rather than an
// anthropic.ContentBlockParamUnion because efs_document is a Pennsieve-
// governor extension, not a standard Anthropic content block type. Pass
// the result as part of the message content list when using
// gov.Client().Messages.New with custom content blocks built from raw
// JSON (or convert to anthropic's RawMessage type as needed).
func EFSDocument(path string) map[string]any {
	return map[string]any{
		"type": "efs_document",
		"path": path,
	}
}

// rawGet sends a SigV4-signed GET against the governor and returns the body.
func (g *Governor) rawGet(ctx context.Context, path string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, g.url+path, nil)
	if err != nil {
		return nil, err
	}
	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}
	if resp.StatusCode >= 400 {
		var errResp ErrorResponse
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil && errResp.Error != "" {
			return nil, &GovernorError{
				Code:          errResp.Error,
				Msg:           errResp.Message,
				AllowedModels: errResp.AllowedModels,
				RetryAfterSec: errResp.RetryAfterSec,
			}
		}
		return nil, fmt.Errorf("governor returned HTTP %d: %s", resp.StatusCode, string(body))
	}
	return body, nil
}
