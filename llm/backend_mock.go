package llm

import (
	"context"
	"fmt"
	"sync"
)

// MockBackend returns canned responses for local testing.
//
// Register responses with SetResponse or SetResponses.
// Unmatched calls return a default echo response.
type MockBackend struct {
	mu        sync.Mutex
	responses []*InvokeResponse
	callLog   []*InvokeRequest
}

// NewMockBackend creates a new mock backend.
func NewMockBackend() *MockBackend {
	return &MockBackend{}
}

// SetResponse sets a single canned response returned for every call.
func (b *MockBackend) SetResponse(resp *InvokeResponse) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.responses = []*InvokeResponse{resp}
}

// SetResponses sets a sequence of responses consumed in order; the last one repeats.
func (b *MockBackend) SetResponses(responses []*InvokeResponse) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.responses = make([]*InvokeResponse, len(responses))
	copy(b.responses, responses)
}

// Calls returns all InvokeRequests received, for test assertions.
func (b *MockBackend) Calls() []*InvokeRequest {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]*InvokeRequest, len(b.callLog))
	copy(out, b.callLog)
	return out
}

func (b *MockBackend) Invoke(_ context.Context, req *InvokeRequest) (*InvokeResponse, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.callLog = append(b.callLog, req)

	if len(b.responses) > 0 {
		if len(b.responses) > 1 {
			resp := b.responses[0]
			b.responses = b.responses[1:]
			return resp, nil
		}
		return b.responses[0], nil
	}

	// Default: echo the last user prompt.
	promptText := ""
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			for _, block := range req.Messages[i].Content {
				if block.Type == "text" && block.Text != "" {
					promptText = block.Text
					break
				}
			}
			break
		}
	}

	return &InvokeResponse{
		Content:    []ResponseContent{{Type: "text", Text: fmt.Sprintf("[mock] %s", promptText)}},
		Model:      req.Model,
		StopReason: "end_turn",
	}, nil
}

func (b *MockBackend) CheckBudget(_ context.Context, _ string) (*CheckBudgetResponse, error) {
	return &CheckBudgetResponse{
		BudgetPeriod:          "mock",
		PeriodBudgetUsd:       100.0,
		PeriodRemainingUsd:    100.0,
		ExecutionBudgetUsd:    10.0,
		ExecutionRemainingUsd: 10.0,
	}, nil
}

func (b *MockBackend) ListModels(_ context.Context) (*ListModelsResponse, error) {
	return &ListModelsResponse{Models: allModels()}, nil
}