package llm

import (
	"bytes"
	"net/http"
	"testing"
)

func TestBuildLambdaEnvelope(t *testing.T) {
	req, _ := http.NewRequest(http.MethodPost, "http://governor.lambda.invoke/v1/messages?foo=bar", bytes.NewBufferString(`{"model":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	// http.Header canonicalizes this to "X-Execution-Run-Id" — same as the
	// sidecar sends; the governor reads it case-insensitively.
	req.Header.Set("x-execution-run-id", "run-123")
	req.Header.Set("Authorization", "Bearer should-be-stripped")
	req.Header.Set("x-api-key", "placeholder-stripped")

	env := buildLambdaEnvelope(req, []byte(`{"model":"x"}`))

	if env.Version != "2.0" {
		t.Errorf("version = %q, want 2.0", env.Version)
	}
	if env.RawPath != "/v1/messages" {
		t.Errorf("rawPath = %q", env.RawPath)
	}
	if env.RawQueryString != "foo=bar" {
		t.Errorf("rawQueryString = %q", env.RawQueryString)
	}
	if env.RequestContext.HTTP.Method != http.MethodPost {
		t.Errorf("method = %q", env.RequestContext.HTTP.Method)
	}
	if env.Body != `{"model":"x"}` || env.IsBase64Encoded {
		t.Errorf("body = %q b64=%v", env.Body, env.IsBase64Encoded)
	}
	if env.Headers["X-Execution-Run-Id"] != "run-123" {
		t.Errorf("expected X-Execution-Run-Id preserved, got %v", env.Headers)
	}
	for _, k := range []string{"Authorization", "x-api-key"} {
		if _, ok := env.Headers[k]; ok {
			t.Errorf("expected %q to be stripped, got %v", k, env.Headers)
		}
	}
}

func TestBuildLambdaEnvelope_Base64Body(t *testing.T) {
	raw := []byte{0xff, 0xfe, 0x00, 0x01} // invalid UTF-8
	req, _ := http.NewRequest(http.MethodPost, "http://x/v1/messages", bytes.NewReader(raw))
	env := buildLambdaEnvelope(req, raw)
	if !env.IsBase64Encoded {
		t.Error("expected non-UTF-8 body to be base64-encoded")
	}
}

func TestSplitPrelude(t *testing.T) {
	body := []byte("response-body-bytes")
	buf := append([]byte(`{"statusCode":200,"headers":{"content-type":"application/json"}}`), preludeSeparator...)
	buf = append(buf, body...)

	pre, leftover, found, err := splitPrelude(buf)
	if err != nil || !found {
		t.Fatalf("found=%v err=%v", found, err)
	}
	if pre.StatusCode != 200 {
		t.Errorf("status = %d", pre.StatusCode)
	}
	if pre.Headers["content-type"] != "application/json" {
		t.Errorf("headers = %v", pre.Headers)
	}
	if string(leftover) != string(body) {
		t.Errorf("leftover = %q, want %q", leftover, body)
	}
}

func TestSplitPrelude_NotYetComplete(t *testing.T) {
	// No separator yet — caller should keep buffering.
	_, _, found, err := splitPrelude([]byte(`{"statusCode":200`))
	if found || err != nil {
		t.Errorf("expected not-found, no error; got found=%v err=%v", found, err)
	}
}

func TestParseWholeEnvelope(t *testing.T) {
	buf := []byte(`{"statusCode":403,"headers":{"content-type":"application/json"},"body":"{\"error\":\"model_not_allowed\"}"}`)
	pre, body, ok := parseWholeEnvelope(buf)
	if !ok {
		t.Fatal("expected ok")
	}
	if pre.StatusCode != 403 {
		t.Errorf("status = %d", pre.StatusCode)
	}
	if string(body) != `{"error":"model_not_allowed"}` {
		t.Errorf("body = %q", body)
	}
}
