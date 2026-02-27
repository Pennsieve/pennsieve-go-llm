package llm

import (
	"context"
	"os"

	"github.com/aws/aws-sdk-go-v2/service/lambda"
)

// Governor is a client for the Pennsieve LLM platform.
type Governor struct {
	functionName   string
	executionRunID string
	lambdaClient   *lambda.Client
	backend        Backend
}

// GovernorOption configures a Governor instance.
type GovernorOption func(*Governor)

// WithFunctionName overrides the governor function name.
// By default, it is read from the LLM_GOVERNOR_FUNCTION env var.
func WithFunctionName(name string) GovernorOption {
	return func(g *Governor) {
		g.functionName = name
	}
}

// WithExecutionRunID sets a default execution run ID for all requests.
// Can be overridden per-request via InvokeRequest.ExecutionRunID.
func WithExecutionRunID(id string) GovernorOption {
	return func(g *Governor) {
		g.executionRunID = id
	}
}

// WithLambdaClient provides a custom Lambda client (useful for testing).
func WithLambdaClient(client *lambda.Client) GovernorOption {
	return func(g *Governor) {
		g.lambdaClient = client
	}
}

// WithBackend provides an explicit backend, overriding automatic selection.
func WithBackend(b Backend) GovernorOption {
	return func(g *Governor) {
		g.backend = b
	}
}

// NewGovernor creates a new Governor client.
//
// Backend is selected automatically based on environment:
//   - If a backend is provided via WithBackend, it is used directly.
//   - If LLM_GOVERNOR_FUNCTION is set (or WithFunctionName is used), a LambdaBackend is used.
//   - If ANTHROPIC_API_KEY is set, an AnthropicBackend is used for local development.
//   - Otherwise, a MockBackend is used for testing.
//
// The AWS Lambda client is created lazily on first use if not provided.
func NewGovernor(opts ...GovernorOption) *Governor {
	g := &Governor{
		functionName:   os.Getenv("LLM_GOVERNOR_FUNCTION"),
		executionRunID: os.Getenv("EXECUTION_RUN_ID"),
	}
	for _, opt := range opts {
		opt(g)
	}

	if g.backend == nil {
		switch {
		case g.functionName != "":
			g.backend = NewLambdaBackend(g.functionName, g.lambdaClient)
		case os.Getenv("ANTHROPIC_API_KEY") != "":
			g.backend = NewAnthropicBackend()
		default:
			g.backend = NewMockBackend()
		}
	}

	return g
}

// Available returns true if the governor is configured with a real backend
// (Lambda or Anthropic). Returns false for the mock backend.
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