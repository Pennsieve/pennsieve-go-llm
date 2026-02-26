package llm

import (
	"context"
	"encoding/json"
	"testing"
)

func TestNewGovernor_Defaults(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "my-governor")
	t.Setenv("EXECUTION_RUN_ID", "run-123")

	g := NewGovernor()
	if g.functionName != "my-governor" {
		t.Errorf("expected functionName 'my-governor', got %q", g.functionName)
	}
	if g.executionRunID != "run-123" {
		t.Errorf("expected executionRunID 'run-123', got %q", g.executionRunID)
	}
	if !g.Available() {
		t.Error("expected Available() to be true")
	}
}

func TestNewGovernor_Options(t *testing.T) {
	g := NewGovernor(
		WithFunctionName("custom-func"),
		WithExecutionRunID("custom-run"),
	)
	if g.functionName != "custom-func" {
		t.Errorf("expected functionName 'custom-func', got %q", g.functionName)
	}
	if g.executionRunID != "custom-run" {
		t.Errorf("expected executionRunID 'custom-run', got %q", g.executionRunID)
	}
}

func TestNewGovernor_NotAvailable(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "")

	g := NewGovernor()
	if g.Available() {
		t.Error("expected Available() to be false when env var is empty")
	}
}

func TestInvokeResponse_Text(t *testing.T) {
	resp := &InvokeResponse{
		Content: []ResponseContent{
			{Type: "text", Text: "Hello "},
			{Type: "text", Text: "world"},
		},
	}
	if resp.Text() != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", resp.Text())
	}
}

func TestInvokeResponse_TextEmpty(t *testing.T) {
	resp := &InvokeResponse{}
	if resp.Text() != "" {
		t.Errorf("expected empty string, got %q", resp.Text())
	}
}

func TestGovernorError(t *testing.T) {
	err := &GovernorError{
		Code: "budget_exceeded",
		Msg:  "daily budget exceeded",
	}

	if err.Error() != "governor error [budget_exceeded]: daily budget exceeded" {
		t.Errorf("unexpected error string: %s", err.Error())
	}
	if !err.IsBudgetExceeded() {
		t.Error("expected IsBudgetExceeded() to be true")
	}
	if err.IsModelNotAllowed() {
		t.Error("expected IsModelNotAllowed() to be false")
	}
}

func TestIsGovernorError(t *testing.T) {
	err := &GovernorError{Code: "model_not_allowed", Msg: "not allowed"}

	ge, ok := IsGovernorError(err)
	if !ok {
		t.Fatal("expected IsGovernorError to return true")
	}
	if !ge.IsModelNotAllowed() {
		t.Error("expected IsModelNotAllowed() to be true")
	}
}

func TestIsGovernorError_NotGovernorError(t *testing.T) {
	_, ok := IsGovernorError(context.DeadlineExceeded)
	if ok {
		t.Error("expected IsGovernorError to return false for non-GovernorError")
	}
}

func TestMessageBuilder(t *testing.T) {
	msg := UserMessage(
		TextBlock("Summarize this"),
		FileBlock("workdir/run-1/output/report.pdf"),
	)
	if msg.Role != "user" {
		t.Errorf("expected role 'user', got %q", msg.Role)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != "text" {
		t.Errorf("expected first block type 'text', got %q", msg.Content[0].Type)
	}
	if msg.Content[1].Type != "efs_document" {
		t.Errorf("expected second block type 'efs_document', got %q", msg.Content[1].Type)
	}
}

func TestInvokeRequest_Serialization(t *testing.T) {
	req := &InvokeRequest{
		Action:         "invoke",
		Model:          ModelHaiku45,
		ExecutionRunID: "run-abc",
		MaxTokens:      512,
		Messages: []Message{
			UserMessage(TextBlock("Hello")),
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var parsed InvokeRequest
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if parsed.Model != ModelHaiku45 {
		t.Errorf("expected model %q, got %q", ModelHaiku45, parsed.Model)
	}
	if parsed.ExecutionRunID != "run-abc" {
		t.Errorf("expected executionRunId 'run-abc', got %q", parsed.ExecutionRunID)
	}
	if len(parsed.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(parsed.Messages))
	}
	if parsed.Messages[0].Content[0].Text != "Hello" {
		t.Errorf("expected text 'Hello', got %q", parsed.Messages[0].Content[0].Text)
	}
}