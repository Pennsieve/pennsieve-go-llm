package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
)

// LambdaBackend calls the LLM Governor Lambda function.
type LambdaBackend struct {
	functionName string
	lambdaClient *lambda.Client
}

// NewLambdaBackend creates a new Lambda backend.
func NewLambdaBackend(functionName string, lambdaClient *lambda.Client) *LambdaBackend {
	return &LambdaBackend{
		functionName: functionName,
		lambdaClient: lambdaClient,
	}
}

func (b *LambdaBackend) ensureClient(ctx context.Context) error {
	if b.lambdaClient != nil {
		return nil
	}
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to load AWS config: %w", err)
	}
	b.lambdaClient = lambda.NewFromConfig(cfg)
	return nil
}

func (b *LambdaBackend) call(ctx context.Context, payload interface{}, result interface{}) error {
	if err := b.ensureClient(ctx); err != nil {
		return err
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	output, err := b.lambdaClient.Invoke(ctx, &lambda.InvokeInput{
		FunctionName: aws.String(b.functionName),
		Payload:      payloadBytes,
	})
	if err != nil {
		return fmt.Errorf("failed to invoke governor: %w", err)
	}

	if output.FunctionError != nil {
		return fmt.Errorf("governor function error: %s", *output.FunctionError)
	}

	// Try to detect a governor error response.
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

func (b *LambdaBackend) Invoke(ctx context.Context, req *InvokeRequest) (*InvokeResponse, error) {
	var resp InvokeResponse
	if err := b.call(ctx, req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (b *LambdaBackend) CheckBudget(ctx context.Context, executionRunID string) (*CheckBudgetResponse, error) {
	payload := map[string]interface{}{
		"action":         "check-budget",
		"executionRunId": executionRunID,
	}
	var resp CheckBudgetResponse
	if err := b.call(ctx, payload, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (b *LambdaBackend) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	payload := map[string]interface{}{
		"action": "list-models",
	}
	var resp ListModelsResponse
	if err := b.call(ctx, payload, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}