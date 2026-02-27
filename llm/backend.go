package llm

import "context"

// Backend is the interface that all LLM backends must satisfy.
type Backend interface {
	Invoke(ctx context.Context, req *InvokeRequest) (*InvokeResponse, error)
	CheckBudget(ctx context.Context, executionRunID string) (*CheckBudgetResponse, error)
	ListModels(ctx context.Context) (*ListModelsResponse, error)
}

// allModels returns ModelInfo for all known models.
func allModels() []ModelInfo {
	return []ModelInfo{
		{ModelID: ModelHaiku45, Status: "available"},
		{ModelID: ModelSonnet45, Status: "available"},
		{ModelID: ModelSonnet46, Status: "available"},
		{ModelID: ModelSonnet4, Status: "available"},
	}
}