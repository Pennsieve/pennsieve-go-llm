package llm

// Types for governor-specific endpoints (CheckBudget, ListModels) and error
// envelopes. Chat request/response types are NOT defined here — those come
// from github.com/anthropics/anthropic-sdk-go, returned via Governor.Client().

// BudgetInfo holds budget tracking data.
type BudgetInfo struct {
	BudgetPeriod          string  `json:"budgetPeriod"`
	PeriodBudgetUsd       float64 `json:"periodBudgetUsd"`
	PeriodUsedUsd         float64 `json:"periodUsedUsd"`
	PeriodRemainingUsd    float64 `json:"periodRemainingUsd"`
	ExecutionBudgetUsd    float64 `json:"executionBudgetUsd,omitempty"`
	ExecutionUsedUsd      float64 `json:"executionUsedUsd,omitempty"`
	ExecutionRemainingUsd float64 `json:"executionRemainingUsd,omitempty"`
}

// CheckBudgetResponse is the response from GET /v1/budget.
type CheckBudgetResponse struct {
	BudgetPeriod          string  `json:"budgetPeriod"`
	PeriodBudgetUsd       float64 `json:"periodBudgetUsd"`
	PeriodUsedUsd         float64 `json:"periodUsedUsd"`
	PeriodRemainingUsd    float64 `json:"periodRemainingUsd"`
	ExecutionBudgetUsd    float64 `json:"executionBudgetUsd,omitempty"`
	ExecutionUsedUsd      float64 `json:"executionUsedUsd,omitempty"`
	ExecutionRemainingUsd float64 `json:"executionRemainingUsd,omitempty"`
}

// ModelInfo represents a model in the ListModels response.
type ModelInfo struct {
	ModelID string `json:"modelId"`
	Status  string `json:"status"`
	Hint    string `json:"hint,omitempty"`
}

// ListModelsResponse is the response from GET /v1/models.
type ListModelsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ErrorResponse is the governor's error envelope (returned by any
// non-Anthropic endpoint like /v1/budget or /v1/models on failure). Chat
// errors come back through the anthropic.Client as anthropic.APIError —
// use anthropic-sdk-go's exception hierarchy for those.
type ErrorResponse struct {
	Error           string      `json:"error"`
	Message         string      `json:"message"`
	AllowedModels   []string    `json:"allowedModels,omitempty"`
	BudgetRemaining *BudgetInfo `json:"budgetRemaining,omitempty"`
	MaxSizeBytes    int64       `json:"maxSizeBytes,omitempty"`
	RetryAfterSec   int         `json:"retryAfterSeconds,omitempty"`
	Model           string      `json:"model,omitempty"`
}
