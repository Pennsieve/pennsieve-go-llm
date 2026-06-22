package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
)

// stubCredentialsProvider returns canned AWS credentials for tests so we
// don't need a real credential chain.
type stubCredentialsProvider struct{}

func (stubCredentialsProvider) Retrieve(_ context.Context) (aws.Credentials, error) {
	return aws.Credentials{
		AccessKeyID:     "AKIA-TEST",
		SecretAccessKey: "secret-test",
		SessionToken:    "session-test",
	}, nil
}

func TestNew_MissingFunctionName(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION_NAME", "")
	_, err := New(context.Background())
	if err == nil {
		t.Error("expected error when no governor function name is configured")
	}
}

func TestNew_WithFunctionName(t *testing.T) {
	t.Setenv("AWS_REGION", "us-east-1")
	t.Setenv("EXECUTION_RUN_ID", "run-test")
	g, err := New(context.Background(),
		WithFunctionName("llm-governor-acct-dev-node1"),
		WithCredentials(stubCredentialsProvider{}),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if g.FunctionName() != "llm-governor-acct-dev-node1" {
		t.Errorf("FunctionName not preserved: %q", g.FunctionName())
	}
	if g.ExecutionRunID() != "run-test" {
		t.Errorf("ExecutionRunID not picked up from env: %q", g.ExecutionRunID())
	}
	if g.Client() == nil {
		t.Error("expected Client() to return non-nil anthropic.Client")
	}
}

func TestNew_FunctionNameFromEnv(t *testing.T) {
	t.Setenv("AWS_REGION", "us-east-1")
	t.Setenv("LLM_GOVERNOR_FUNCTION_NAME", "llm-governor-from-env")
	g, err := New(context.Background(), WithCredentials(stubCredentialsProvider{}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if g.FunctionName() != "llm-governor-from-env" {
		t.Errorf("FunctionName not read from env: %q", g.FunctionName())
	}
}

func TestCheckBudget_AgainstFakeServer(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/budget" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"budgetPeriod": "daily",
			"periodBudgetUsd": 5.0,
			"periodUsedUsd": 1.25,
			"periodRemainingUsd": 3.75
		}`))
	}))
	defer ts.Close()

	g, err := New(context.Background(),
		WithBaseURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	b, err := g.CheckBudget(context.Background())
	if err != nil {
		t.Fatalf("CheckBudget failed: %v", err)
	}
	if b.BudgetPeriod != "daily" || b.PeriodUsedUsd != 1.25 || b.PeriodRemainingUsd != 3.75 {
		t.Errorf("CheckBudget response not parsed correctly: %+v", b)
	}
}

func TestCheckBudget_ErrorResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"invalid_request","message":"bad input"}`))
	}))
	defer ts.Close()

	g, err := New(context.Background(),
		WithBaseURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = g.CheckBudget(context.Background())
	if err == nil {
		t.Fatal("expected error from 400 response")
	}
	ge, ok := IsGovernorError(err)
	if !ok {
		t.Errorf("expected GovernorError, got %T: %v", err, err)
	}
	if ge.Code != "invalid_request" {
		t.Errorf("expected code=invalid_request, got %q", ge.Code)
	}
}

func TestListModels_AgainstFakeServer(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"models": [
				{"modelId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "status": "available"},
				{"modelId": "us.anthropic.claude-opus-4-7", "status": "available"}
			]
		}`))
	}))
	defer ts.Close()

	g, err := New(context.Background(),
		WithBaseURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp, err := g.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}
	if len(resp.Models) != 2 {
		t.Errorf("expected 2 models, got %d", len(resp.Models))
	}
}

func TestEFSDocument(t *testing.T) {
	block := EFSDocument("workdir/paper.pdf")
	if block["type"] != "efs_document" {
		t.Errorf("type should be efs_document, got %v", block["type"])
	}
	if block["path"] != "workdir/paper.pdf" {
		t.Errorf("path should be workdir/paper.pdf, got %v", block["path"])
	}
}

func TestGovernorError_Predicates(t *testing.T) {
	err := &GovernorError{Code: "budget_exceeded", Msg: "over"}
	if !err.IsBudgetExceeded() {
		t.Error("expected IsBudgetExceeded() to be true")
	}
	if err.IsModelNotAllowed() {
		t.Error("expected IsModelNotAllowed() to be false")
	}
}

func TestIsGovernorError(t *testing.T) {
	plain := http.ErrAbortHandler
	if _, ok := IsGovernorError(plain); ok {
		t.Error("non-GovernorError should not match")
	}
	gov := &GovernorError{Code: "x", Msg: "y"}
	got, ok := IsGovernorError(gov)
	if !ok || got != gov {
		t.Errorf("expected to match, got %v %v", got, ok)
	}
}

func TestModelConstants(t *testing.T) {
	models := []string{ModelHaiku45, ModelSonnet4, ModelSonnet45, ModelSonnet46, ModelOpus47}
	for _, m := range models {
		if len(m) < 13 || m[:13] != "us.anthropic." {
			t.Errorf("model %q should start with us.anthropic.", m)
		}
	}
}
