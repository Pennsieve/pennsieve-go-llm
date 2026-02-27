package llm

import (
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"
)

// --- Model mapping tests ---

func TestMapModel_KnownModels(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{ModelHaiku45, "claude-haiku-4-5-20251001"},
		{ModelSonnet45, "claude-sonnet-4-5-20250929"},
		{ModelSonnet46, "claude-sonnet-4-6"},
		{ModelSonnet4, "claude-sonnet-4-20250514"},
	}
	for _, tt := range tests {
		got := MapModel(tt.input)
		if got != tt.want {
			t.Errorf("MapModel(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestMapModel_FallbackStripsPrefix(t *testing.T) {
	got := MapModel("us.anthropic.claude-future-model")
	if got != "claude-future-model" {
		t.Errorf("expected 'claude-future-model', got %q", got)
	}
}

func TestMapModel_PassthroughUnknown(t *testing.T) {
	got := MapModel("some-other-model")
	if got != "some-other-model" {
		t.Errorf("expected 'some-other-model', got %q", got)
	}
}

// --- Message conversion tests ---

func TestConvertMessages_TextMessage(t *testing.T) {
	msgs := []Message{
		UserMessage(TextBlock("Hello")),
	}
	result := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0]["role"] != "user" {
		t.Errorf("expected role 'user', got %v", result[0]["role"])
	}
	blocks := result[0]["content"].([]interface{})
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	block := blocks[0].(map[string]interface{})
	if block["type"] != "text" {
		t.Errorf("expected type 'text', got %v", block["type"])
	}
	if block["text"] != "Hello" {
		t.Errorf("expected text 'Hello', got %v", block["text"])
	}
}

func TestConvertMessages_ImageMessage(t *testing.T) {
	msgs := []Message{
		UserMessage(ImageBlock("png", "base64data")),
	}
	result := convertMessages(msgs)
	blocks := result[0]["content"].([]interface{})
	block := blocks[0].(map[string]interface{})
	if block["type"] != "image" {
		t.Errorf("expected type 'image', got %v", block["type"])
	}
	source := block["source"].(map[string]interface{})
	if source["type"] != "base64" {
		t.Errorf("expected source type 'base64', got %v", source["type"])
	}
	if source["media_type"] != "image/png" {
		t.Errorf("expected media_type 'image/png', got %v", source["media_type"])
	}
	if source["data"] != "base64data" {
		t.Errorf("expected data 'base64data', got %v", source["data"])
	}
}

func TestConvertMessages_EFSDocumentMissing(t *testing.T) {
	msgs := []Message{
		UserMessage(FileBlock("/nonexistent/file.pdf")),
	}
	result := convertMessages(msgs)
	blocks := result[0]["content"].([]interface{})
	block := blocks[0].(map[string]interface{})
	if block["type"] != "text" {
		t.Errorf("expected type 'text' for missing file, got %v", block["type"])
	}
	text := block["text"].(string)
	if text != "[File not available locally: /nonexistent/file.pdf]" {
		t.Errorf("unexpected fallback text: %s", text)
	}
}

func TestConvertMessages_EFSDocumentExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pdf")
	content := []byte("fake pdf content")
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatal(err)
	}

	msgs := []Message{
		UserMessage(ContentBlock{Type: "efs_document", Path: path}),
	}
	result := convertMessages(msgs)
	blocks := result[0]["content"].([]interface{})
	block := blocks[0].(map[string]interface{})
	if block["type"] != "document" {
		t.Errorf("expected type 'document', got %v", block["type"])
	}
	source := block["source"].(map[string]interface{})
	if source["media_type"] != "application/pdf" {
		t.Errorf("expected media_type 'application/pdf', got %v", source["media_type"])
	}
	decoded, err := base64.StdEncoding.DecodeString(source["data"].(string))
	if err != nil {
		t.Fatalf("failed to decode base64: %v", err)
	}
	if string(decoded) != "fake pdf content" {
		t.Errorf("expected decoded content 'fake pdf content', got %q", string(decoded))
	}
}

func TestConvertMessages_DocumentBlock(t *testing.T) {
	msgs := []Message{
		UserMessage(DocumentBlock("report", "pdf", "base64pdf")),
	}
	result := convertMessages(msgs)
	blocks := result[0]["content"].([]interface{})
	block := blocks[0].(map[string]interface{})
	if block["type"] != "document" {
		t.Errorf("expected type 'document', got %v", block["type"])
	}
	source := block["source"].(map[string]interface{})
	if source["media_type"] != "application/pdf" {
		t.Errorf("expected media_type 'application/pdf', got %v", source["media_type"])
	}
}

func TestConvertMessages_UnsupportedBlock(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: []ContentBlock{{Type: "unknown_type"}}},
	}
	result := convertMessages(msgs)
	blocks := result[0]["content"].([]interface{})
	block := blocks[0].(map[string]interface{})
	if block["type"] != "text" {
		t.Errorf("expected fallback type 'text', got %v", block["type"])
	}
	if block["text"] != "[Unsupported content block: unknown_type]" {
		t.Errorf("unexpected fallback text: %v", block["text"])
	}
}

// --- MockBackend tests ---

func TestMockBackend_DefaultEcho(t *testing.T) {
	b := NewMockBackend()
	resp, err := b.Invoke(context.Background(), &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("Hello world"))},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "[mock] Hello world" {
		t.Errorf("expected '[mock] Hello world', got %q", resp.Text())
	}
	if resp.Model != ModelHaiku45 {
		t.Errorf("expected model %q, got %q", ModelHaiku45, resp.Model)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", resp.StopReason)
	}
}

func TestMockBackend_SetResponse(t *testing.T) {
	b := NewMockBackend()
	canned := &InvokeResponse{
		Content: []ResponseContent{{Type: "text", Text: "canned response"}},
		Model:   "test-model",
	}
	b.SetResponse(canned)

	resp, err := b.Invoke(context.Background(), &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("anything"))},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "canned response" {
		t.Errorf("expected 'canned response', got %q", resp.Text())
	}

	// Second call returns the same response.
	resp2, _ := b.Invoke(context.Background(), &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("anything else"))},
	})
	if resp2.Text() != "canned response" {
		t.Errorf("expected 'canned response' again, got %q", resp2.Text())
	}
}

func TestMockBackend_SetResponses(t *testing.T) {
	b := NewMockBackend()
	b.SetResponses([]*InvokeResponse{
		{Content: []ResponseContent{{Type: "text", Text: "first"}}},
		{Content: []ResponseContent{{Type: "text", Text: "second"}}},
	})

	resp1, _ := b.Invoke(context.Background(), &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("a"))},
	})
	if resp1.Text() != "first" {
		t.Errorf("expected 'first', got %q", resp1.Text())
	}

	resp2, _ := b.Invoke(context.Background(), &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("b"))},
	})
	if resp2.Text() != "second" {
		t.Errorf("expected 'second', got %q", resp2.Text())
	}

	// Last response repeats.
	resp3, _ := b.Invoke(context.Background(), &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("c"))},
	})
	if resp3.Text() != "second" {
		t.Errorf("expected 'second' (repeat), got %q", resp3.Text())
	}
}

func TestMockBackend_CallLog(t *testing.T) {
	b := NewMockBackend()
	req := &InvokeRequest{
		Model:    ModelHaiku45,
		Messages: []Message{UserMessage(TextBlock("log me"))},
	}
	b.Invoke(context.Background(), req)

	calls := b.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Messages[0].Content[0].Text != "log me" {
		t.Errorf("expected logged text 'log me', got %q", calls[0].Messages[0].Content[0].Text)
	}
}

func TestMockBackend_CheckBudget(t *testing.T) {
	b := NewMockBackend()
	resp, err := b.CheckBudget(context.Background(), "run-1")
	if err != nil {
		t.Fatal(err)
	}
	if resp.BudgetPeriod != "mock" {
		t.Errorf("expected budget period 'mock', got %q", resp.BudgetPeriod)
	}
	if resp.PeriodBudgetUsd != 100.0 {
		t.Errorf("expected period budget 100.0, got %f", resp.PeriodBudgetUsd)
	}
}

func TestMockBackend_ListModels(t *testing.T) {
	b := NewMockBackend()
	resp, err := b.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Models) != 4 {
		t.Errorf("expected 4 models, got %d", len(resp.Models))
	}
}

// --- Backend selection tests ---

func TestGovernor_NoEnvSelectsMock(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	g := NewGovernor()
	if _, ok := g.backend.(*MockBackend); !ok {
		t.Errorf("expected MockBackend, got %T", g.backend)
	}
	if g.Available() {
		t.Error("expected Available() to be false for MockBackend")
	}
}

func TestGovernor_LambdaEnvSelectsLambda(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "my-func")
	t.Setenv("ANTHROPIC_API_KEY", "")

	g := NewGovernor()
	if _, ok := g.backend.(*LambdaBackend); !ok {
		t.Errorf("expected LambdaBackend, got %T", g.backend)
	}
	if !g.Available() {
		t.Error("expected Available() to be true for LambdaBackend")
	}
}

func TestGovernor_FunctionNameArgSelectsLambda(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")

	g := NewGovernor(WithFunctionName("explicit-func"))
	if _, ok := g.backend.(*LambdaBackend); !ok {
		t.Errorf("expected LambdaBackend, got %T", g.backend)
	}
}

func TestGovernor_AnthropicEnvSelectsAnthropic(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")

	g := NewGovernor()
	if _, ok := g.backend.(*AnthropicBackend); !ok {
		t.Errorf("expected AnthropicBackend, got %T", g.backend)
	}
	if !g.Available() {
		t.Error("expected Available() to be true for AnthropicBackend")
	}
}

func TestGovernor_ExplicitBackendOverridesEnv(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "my-func")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")

	mock := NewMockBackend()
	g := NewGovernor(WithBackend(mock))
	if g.backend != mock {
		t.Errorf("expected explicit backend to be used, got %T", g.backend)
	}
}

func TestGovernor_MockBackendAsk(t *testing.T) {
	t.Setenv("LLM_GOVERNOR_FUNCTION", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	g := NewGovernor()
	text, err := g.Ask(context.Background(), ModelHaiku45, "Hello")
	if err != nil {
		t.Fatal(err)
	}
	if text != "[mock] Hello" {
		t.Errorf("expected '[mock] Hello', got %q", text)
	}
}

func TestGovernor_BackendAccessor(t *testing.T) {
	mock := NewMockBackend()
	g := NewGovernor(WithBackend(mock))
	if g.Backend() != mock {
		t.Error("Backend() should return the active backend")
	}
}