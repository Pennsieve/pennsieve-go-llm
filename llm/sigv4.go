package llm

// SigV4 http.RoundTripper. Signs every outgoing request against the AWS
// "lambda" service (the signing name for Lambda Function URLs). Wraps an
// inner transport (typically http.DefaultTransport).

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
)

// sigV4Transport is an http.RoundTripper that SigV4-signs every outgoing
// request before delegating to a wrapped transport.
type sigV4Transport struct {
	wrapped http.RoundTripper
	creds   aws.CredentialsProvider
	region  string
}

func (t *sigV4Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Read and replace the body so we can both hash it and pass it down.
	var bodyBytes []byte
	if req.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("read request body for SigV4: %w", err)
		}
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	}

	hash := sha256.Sum256(bodyBytes)
	payloadHash := hex.EncodeToString(hash[:])

	creds, err := t.creds.Retrieve(req.Context())
	if err != nil {
		return nil, fmt.Errorf("retrieve AWS credentials: %w", err)
	}

	signer := v4.NewSigner()
	if err := signer.SignHTTP(req.Context(), creds, req, payloadHash, "lambda", t.region, time.Now()); err != nil {
		return nil, fmt.Errorf("SigV4 sign: %w", err)
	}

	wrapped := t.wrapped
	if wrapped == nil {
		wrapped = http.DefaultTransport
	}
	return wrapped.RoundTrip(req)
}

// newSigV4HTTPClient builds an http.Client whose Transport signs every
// request via SigV4 for the "lambda" service. Used by Governor to build
// the underlying anthropic.Client's HTTP client.
//
// The ctx is only used to retrieve credentials at construction time —
// per-request credential refresh happens automatically via the
// CredentialsProvider.
func newSigV4HTTPClient(ctx context.Context, creds aws.CredentialsProvider, region string, timeout time.Duration) *http.Client {
	_ = ctx // reserved for future use; CredentialsProvider handles refresh
	return &http.Client{
		Transport: &sigV4Transport{
			wrapped: http.DefaultTransport,
			creds:   creds,
			region:  region,
		},
		Timeout: timeout,
	}
}
