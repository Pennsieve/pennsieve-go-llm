package llm

import (
	"context"
	"net/http"
	"os"
)

// Governor is a client for the Pennsieve LLM platform.
type Governor struct {
	governorURL    string
	executionRunID string
	httpClient     *http.Client
	backend        Backend
}

// GovernorOption configures a Governor instance.
type GovernorOption func(*Governor)

// WithURL overrides the governor URL. By default it is read from the
// LLM_GOVERNOR_URL env var.
func WithURL(u string) GovernorOption {
	return func(g *Governor) { g.governorURL = u }
}

// WithExecutionRunID sets a default execution run ID for all requests.
// Can be overridden per-request via InvokeRequest.ExecutionRunID.
func WithExecutionRunID(id string) GovernorOption {
	return func(g *Governor) { g.executionRunID = id }
}

// WithGovernorHTTPClientOption (Governor-level) provides a custom http.Client
// for the governor backend (useful for testing or for tuned timeouts).
// Distinct from WithGovernorHTTPClient which is a backend-level option.
func WithGovernorHTTPClientOption(c *http.Client) GovernorOption {
	return func(g *Governor) { g.httpClient = c }
}

// WithBackend provides an explicit backend, overriding automatic selection.
func WithBackend(b Backend) GovernorOption {
	return func(g *Governor) { g.backend = b }
}

// NewGovernor creates a new Governor client.
//
// Backend is selected automatically based on environment:
//   - If a backend is provided via WithBackend, it is used directly.
//   - If LLM_GOVERNOR_URL is set (or WithURL is used), a GovernorBackend is
//     used. AWS credentials are loaded from the default chain for SigV4.
//   - If ANTHROPIC_API_KEY is set, an AnthropicBackend is used for local
//     development against api.anthropic.com.
//   - Otherwise, a MockBackend is used for testing.
//
// If GovernorBackend setup fails (e.g. AWS config cannot load), the
// constructor falls back to MockBackend rather than panicking. Callers
// can check g.Available() to detect this.
func NewGovernor(opts ...GovernorOption) *Governor {
	g := &Governor{
		governorURL:    os.Getenv("LLM_GOVERNOR_URL"),
		executionRunID: os.Getenv("EXECUTION_RUN_ID"),
	}
	for _, opt := range opts {
		opt(g)
	}

	if g.backend == nil {
		switch {
		case g.governorURL != "":
			govOpts := []GovernorBackendOption{WithGovernorURL(g.governorURL)}
			if g.httpClient != nil {
				govOpts = append(govOpts, WithGovernorHTTPClient(g.httpClient))
			}
			b, err := NewGovernorBackend(context.Background(), govOpts...)
			if err != nil {
				// Fall back to mock so callers can still construct a Governor
				// in environments where AWS config isn't available. They can
				// detect this via g.Available() == false.
				g.backend = NewMockBackend()
			} else {
				g.backend = b
			}
		case os.Getenv("ANTHROPIC_API_KEY") != "":
			g.backend = NewAnthropicBackend()
		default:
			g.backend = NewMockBackend()
		}
	}

	return g
}

// Available returns true if the governor is configured with a real backend
// (governor or direct Anthropic). Returns false for the mock backend.
func (g *Governor) Available() bool {
	_, isMock := g.backend.(*MockBackend)
	return !isMock
}

// Backend returns the active backend instance.
func (g *Governor) Backend() Backend {
	return g.backend
}

// Invoke sends messages to a model and returns the response.
func (g *Governor) Invoke(ctx context.Context, req *InvokeRequest) (*InvokeResponse, error) {
	if req.Action == "" {
		req.Action = "invoke"
	}
	if req.ExecutionRunID == "" {
		req.ExecutionRunID = g.executionRunID
	}
	return g.backend.Invoke(ctx, req)
}

// Ask is a convenience method for simple text-in, text-out interactions.
func (g *Governor) Ask(ctx context.Context, model, prompt string) (string, error) {
	resp, err := g.Invoke(ctx, &InvokeRequest{
		Model:    model,
		Messages: []Message{UserMessage(TextBlock(prompt))},
	})
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// AskWithSystem is like Ask but includes a system prompt.
func (g *Governor) AskWithSystem(ctx context.Context, model, system, prompt string) (string, error) {
	resp, err := g.Invoke(ctx, &InvokeRequest{
		Model:    model,
		System:   system,
		Messages: []Message{UserMessage(TextBlock(prompt))},
	})
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// AskAboutFile sends a text prompt along with an EFS file to the model.
func (g *Governor) AskAboutFile(ctx context.Context, model, prompt, filePath string) (string, error) {
	resp, err := g.Invoke(ctx, &InvokeRequest{
		Model: model,
		Messages: []Message{
			UserMessage(TextBlock(prompt), FileBlock(filePath)),
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// CheckBudget returns the current budget status.
func (g *Governor) CheckBudget(ctx context.Context) (*CheckBudgetResponse, error) {
	return g.backend.CheckBudget(ctx, g.executionRunID)
}

// ListModels returns the available models and their status.
func (g *Governor) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	return g.backend.ListModels(ctx)
}

// Text returns the concatenated text content from the response.
func (r *InvokeResponse) Text() string {
	var text string
	for _, c := range r.Content {
		if c.Type == "text" {
			text += c.Text
		}
	}
	return text
}
