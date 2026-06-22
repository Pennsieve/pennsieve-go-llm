package llm

// lambdaInvokeTransport dispatches each outgoing HTTP request through the AWS
// Lambda API (lambda:InvokeWithResponseStream) instead of over the network,
// wrapping the request as a LambdaFunctionURLRequest envelope. This is how the
// Pennsieve SDK reaches the LLM Governor: there is no public Function URL, so
// all governor traffic stays on private AWS APIs (and works in isolated/no-
// internet compliant VPCs via the Lambda VPC endpoint).
//
// The envelope shape and stream framing mirror the platform's llm-governor-proxy
// sidecar (cmd/llm-governor-proxy in compute-node-aws-provisioner) so the
// governor's lambdaurl.Start(mux) entrypoint handles SDK and sidecar traffic
// identically.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
	lambdatypes "github.com/aws/aws-sdk-go-v2/service/lambda/types"
)

// preludeSeparator is the 8-byte null terminator Lambda response streaming
// emits between the {statusCode, headers} prelude and the response body.
var preludeSeparator = []byte{0, 0, 0, 0, 0, 0, 0, 0}

// strippedRequestHeaders are transport-managed (or auth) headers that don't
// belong on the far side of a lambda:Invoke envelope. Mirrors the sidecar's
// set: the governor authenticates via the IAM identity of the Invoke call, so
// any client-side Authorization / x-api-key is meaningless and dropped.
var strippedRequestHeaders = map[string]struct{}{
	"host": {}, "authorization": {}, "x-api-key": {},
	"anthropic-version": {}, "anthropic-beta": {},
	"anthropic-dangerous-direct-browser-access": {},
	"content-length": {}, "accept-encoding": {}, "connection": {},
	"proxy-connection": {}, "keep-alive": {}, "upgrade": {}, "te": {},
	"trailer": {}, "transfer-encoding": {},
}

// lambdaInvoker is the subset of the AWS Lambda client the transport needs.
// Declared as an interface so the resolution logic stays decoupled from the
// concrete client.
type lambdaInvoker interface {
	InvokeWithResponseStream(ctx context.Context, in *lambda.InvokeWithResponseStreamInput, optFns ...func(*lambda.Options)) (*lambda.InvokeWithResponseStreamOutput, error)
}

// lambdaInvokeTransport is an http.RoundTripper backed by lambda:Invoke.
type lambdaInvokeTransport struct {
	client       lambdaInvoker
	functionName string
}

// lambdaFunctionURLRequest is the v2.0 Function-URL event envelope.
type lambdaFunctionURLRequest struct {
	Version         string                          `json:"version"`
	RawPath         string                          `json:"rawPath"`
	RawQueryString  string                          `json:"rawQueryString"`
	Headers         map[string]string               `json:"headers"`
	RequestContext  lambdaFunctionURLRequestContext `json:"requestContext"`
	Body            string                          `json:"body"`
	IsBase64Encoded bool                            `json:"isBase64Encoded"`
}

type lambdaFunctionURLRequestContext struct {
	RequestID string                       `json:"requestId"`
	HTTP      lambdaFunctionURLHTTPContext `json:"http"`
}

type lambdaFunctionURLHTTPContext struct {
	Method    string `json:"method"`
	Path      string `json:"path"`
	Protocol  string `json:"protocol"`
	SourceIP  string `json:"sourceIp"`
	UserAgent string `json:"userAgent"`
}

// streamPrelude is the {statusCode, headers} blob Lambda response streaming
// prepends to the body.
type streamPrelude struct {
	StatusCode int               `json:"statusCode"`
	Headers    map[string]string `json:"headers"`
}

func (t *lambdaInvokeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		b, err := io.ReadAll(req.Body)
		_ = req.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("read request body: %w", err)
		}
		body = b
	}

	payload, err := json.Marshal(buildLambdaEnvelope(req, body))
	if err != nil {
		return nil, fmt.Errorf("marshal lambda envelope: %w", err)
	}

	out, err := t.client.InvokeWithResponseStream(req.Context(), &lambda.InvokeWithResponseStreamInput{
		FunctionName: aws.String(t.functionName),
		Payload:      payload,
	})
	if err != nil {
		return nil, fmt.Errorf("lambda invoke: %w", err)
	}

	stream := out.GetStream()
	events := stream.Events()

	pre, leftover, err := readPreludeFromEvents(events, stream)
	if err != nil {
		_ = stream.Close()
		return nil, err
	}

	status := pre.StatusCode
	if status == 0 {
		status = http.StatusOK
	}
	header := make(http.Header, len(pre.Headers))
	for k, v := range pre.Headers {
		header.Set(k, v)
	}

	return &http.Response{
		StatusCode: status,
		Status:     fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     header,
		Body:       &lambdaStreamBody{stream: stream, events: events, pending: leftover},
		Request:    req,
	}, nil
}

// buildLambdaEnvelope wraps an HTTP request as a Function-URL event. Pure
// (no AWS calls) so it is unit-testable.
func buildLambdaEnvelope(req *http.Request, body []byte) lambdaFunctionURLRequest {
	headers := make(map[string]string, len(req.Header))
	for k, vs := range req.Header {
		if _, drop := strippedRequestHeaders[strings.ToLower(k)]; drop {
			continue
		}
		if len(vs) > 0 {
			headers[k] = vs[0]
		}
	}

	bodyStr, isB64 := encodeBody(body)
	path := req.URL.Path
	if path == "" {
		path = "/"
	}
	return lambdaFunctionURLRequest{
		Version:         "2.0",
		RawPath:         path,
		RawQueryString:  req.URL.RawQuery,
		Headers:         headers,
		Body:            bodyStr,
		IsBase64Encoded: isB64,
		RequestContext: lambdaFunctionURLRequestContext{
			RequestID: "sdk-invoke",
			HTTP: lambdaFunctionURLHTTPContext{
				Method:    req.Method,
				Path:      path,
				Protocol:  "HTTP/1.1",
				SourceIP:  "127.0.0.1",
				UserAgent: firstNonEmpty(headers["User-Agent"], "pennsieve-go-llm"),
			},
		},
	}
}

// encodeBody returns (body, isBase64). Non-UTF-8 bodies are base64-encoded so
// the JSON envelope stays well-formed.
func encodeBody(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}
	if utf8.Valid(body) {
		return string(body), false
	}
	return base64.StdEncoding.EncodeToString(body), true
}

func firstNonEmpty(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// splitPrelude looks for the prelude separator in buf. When present it returns
// the parsed prelude, any body bytes that followed the separator, and ok=true.
// Pure and unit-testable.
func splitPrelude(buf []byte) (streamPrelude, []byte, bool, error) {
	idx := bytes.Index(buf, preludeSeparator)
	if idx < 0 {
		return streamPrelude{}, nil, false, nil
	}
	var pre streamPrelude
	if err := json.Unmarshal(buf[:idx], &pre); err != nil {
		return streamPrelude{}, nil, false, fmt.Errorf("parse prelude json: %w", err)
	}
	leftover := append([]byte(nil), buf[idx+len(preludeSeparator):]...)
	return pre, leftover, true, nil
}

// parseWholeEnvelope is the fallback when the stream ends without a separator
// (a non-streamed response that fit in a single message). It parses the buffer
// as a full Function-URL response, preserving the body.
func parseWholeEnvelope(buf []byte) (streamPrelude, []byte, bool) {
	var full struct {
		StatusCode      int               `json:"statusCode"`
		Headers         map[string]string `json:"headers"`
		Body            string            `json:"body"`
		IsBase64Encoded bool              `json:"isBase64Encoded"`
	}
	if err := json.Unmarshal(buf, &full); err != nil || (full.StatusCode == 0 && full.Headers == nil) {
		return streamPrelude{}, nil, false
	}
	bodyBytes := []byte(full.Body)
	if full.IsBase64Encoded {
		if decoded, derr := base64.StdEncoding.DecodeString(full.Body); derr == nil {
			bodyBytes = decoded
		}
	}
	return streamPrelude{StatusCode: full.StatusCode, Headers: full.Headers}, bodyBytes, true
}

// readPreludeFromEvents drains chunks from the event channel until the prelude
// is recovered, leaving any subsequent body chunks on the channel for the
// response Body to stream.
func readPreludeFromEvents(events <-chan lambdatypes.InvokeWithResponseStreamResponseEvent, stream *lambda.InvokeWithResponseStreamEventStream) (streamPrelude, []byte, error) {
	var buf []byte
	for ev := range events {
		switch v := ev.(type) {
		case *lambdatypes.InvokeWithResponseStreamResponseEventMemberPayloadChunk:
			buf = append(buf, v.Value.Payload...)
			if pre, leftover, found, err := splitPrelude(buf); err != nil {
				return streamPrelude{}, nil, err
			} else if found {
				return pre, leftover, nil
			}
		case *lambdatypes.InvokeWithResponseStreamResponseEventMemberInvokeComplete:
			if v.Value.ErrorCode != nil && aws.ToString(v.Value.ErrorCode) != "" {
				return streamPrelude{}, nil, fmt.Errorf("lambda invoke error: %s: %s",
					aws.ToString(v.Value.ErrorCode), aws.ToString(v.Value.ErrorDetails))
			}
			if pre, leftover, ok := parseWholeEnvelope(buf); ok {
				return pre, leftover, nil
			}
			return streamPrelude{StatusCode: http.StatusOK}, buf, nil
		}
	}
	if err := stream.Err(); err != nil {
		return streamPrelude{}, nil, fmt.Errorf("lambda stream: %w", err)
	}
	return streamPrelude{}, nil, fmt.Errorf("lambda stream ended before prelude separator")
}

// lambdaStreamBody is the http.Response body backed by the remaining Lambda
// event stream. It streams chunks lazily so the Anthropic SDK can consume SSE.
type lambdaStreamBody struct {
	stream    *lambda.InvokeWithResponseStreamEventStream
	events    <-chan lambdatypes.InvokeWithResponseStreamResponseEvent
	pending   []byte
	finished  bool
	streamErr error
}

func (b *lambdaStreamBody) Read(p []byte) (int, error) {
	for len(b.pending) == 0 && !b.finished {
		ev, ok := <-b.events
		if !ok {
			b.finished = true
			if err := b.stream.Err(); err != nil {
				b.streamErr = fmt.Errorf("lambda stream: %w", err)
			}
			break
		}
		switch v := ev.(type) {
		case *lambdatypes.InvokeWithResponseStreamResponseEventMemberPayloadChunk:
			b.pending = append(b.pending, v.Value.Payload...)
		case *lambdatypes.InvokeWithResponseStreamResponseEventMemberInvokeComplete:
			if v.Value.ErrorCode != nil && aws.ToString(v.Value.ErrorCode) != "" {
				b.finished = true
				b.streamErr = fmt.Errorf("lambda invoke error mid-stream: %s: %s",
					aws.ToString(v.Value.ErrorCode), aws.ToString(v.Value.ErrorDetails))
			}
		}
	}
	if len(b.pending) > 0 {
		n := copy(p, b.pending)
		b.pending = b.pending[n:]
		return n, nil
	}
	if b.streamErr != nil {
		return 0, b.streamErr
	}
	return 0, io.EOF
}

func (b *lambdaStreamBody) Close() error {
	if b.stream != nil {
		return b.stream.Close()
	}
	return nil
}
