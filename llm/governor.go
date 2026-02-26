package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
)

// Governor is a client for the LLM Governor Lambda.
type Governor struct {
	functionName   string
	executionRunID string
	lambdaClient   *lambda.Client
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

// NewGovernor creates a new Governor client.
//
// By default it reads the function name from LLM_GOVERNOR_FUNCTION and the
// execution run ID from EXECUTION_RUN_ID environment variables. Both can be
// overridden with options.
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
	return g
}

// ensureClient lazily initializes the Lambda client.
func (g *Governor) ensureClient(ctx context.Context) error {
	if g.lambdaClient != nil {
		return nil
	}
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to load AWS config: %w", err)
	}
	g.lambdaClient = lambda.NewFromConfig(cfg)
	return nil
}

// Available returns true if the governor function is configured.
// Processors should check this before attempting LLM calls.
func (g *Governor) Available() bool {
	return g.functionName != ""
}

// invoke calls the governor Lambda with the given payload and unmarshals the response.
func (g *Governor) invoke(ctx context.Context, payload interface{}, result interface{}) error {
	if !g.Available() {
		return fmt.Errorf("LLM governor not available: LLM_GOVERNOR_FUNCTION is not set")
	}

	if err := g.ensureClient(ctx); err != nil {
		return err
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	output, err := g.lambdaClient.Invoke(ctx, &lambda.InvokeInput{
		FunctionName: aws.String(g.functionName),
		Payload:      payloadBytes,
	})
	if err != nil {
		return fmt.Errorf("failed to invoke governor: %w", err)
	}

	if output.FunctionError != nil {
		return fmt.Errorf("governor function error: %s", *output.FunctionError)
	}

	// Try to detect a governor error response
	var errResp ErrorResponse
	if err := json.Unmarshal(output.Payload, &errResp); err == nil && errResp.Error != "" {
		return &GovernorError{
			Code:            errResp.Error,
			Msg:             errResp.Message,
			AllowedModels:   errResp.AllowedModels,
			BudgetRemaining: errResp.BudgetRemaining,
			RetryAfterSec:   errResp.RetryAfterSec,
		}
	}

	if err := json.Unmarshal(output.Payload, result); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return nil
}

// Invoke sends messages to a model and returns the response.
func (g *Governor) Invoke(ctx context.Context, req *InvokeRequest) (*InvokeResponse, error) {
	if req.Action == "" {
		req.Action = "invoke"
	}
	if req.ExecutionRunID == "" {
		req.ExecutionRunID = g.executionRunID
	}
	if req.ExecutionRunID == "" {
		return nil, fmt.Errorf("executionRunId is required: set EXECUTION_RUN_ID env var or use WithExecutionRunID option")
	}

	var resp InvokeResponse
	if err := g.invoke(ctx, req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Ask is a convenience method for simple text-in, text-out interactions.
func (g *Governor) Ask(ctx context.Context, model, prompt string) (string, error) {
	resp, err := g.Invoke(ctx, &InvokeRequest{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: []ContentBlock{{Type: "text", Text: prompt}}},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// AskWithSystem is like Ask but includes a system prompt.
func (g *Governor) AskWithSystem(ctx context.Context, model, system, prompt string) (string, error) {
	resp, err := g.Invoke(ctx, &InvokeRequest{
		Model:  model,
		System: system,
		Messages: []Message{
			{Role: "user", Content: []ContentBlock{{Type: "text", Text: prompt}}},
		},
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
			{Role: "user", Content: []ContentBlock{
				{Type: "text", Text: prompt},
				{Type: "efs_document", Path: filePath},
			}},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// CheckBudget returns the current budget status without making a Bedrock call.
func (g *Governor) CheckBudget(ctx context.Context) (*CheckBudgetResponse, error) {
	req := map[string]interface{}{
		"action":         "check-budget",
		"executionRunId": g.executionRunID,
	}
	var resp CheckBudgetResponse
	if err := g.invoke(ctx, req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListModels returns the available models and their status.
func (g *Governor) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	req := map[string]interface{}{
		"action": "list-models",
	}
	var resp ListModelsResponse
	if err := g.invoke(ctx, req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
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