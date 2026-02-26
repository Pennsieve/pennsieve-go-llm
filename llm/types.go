package llm

// InvokeRequest is the request payload for an LLM invocation.
type InvokeRequest struct {
	Action             string    `json:"action"`
	Model              string    `json:"model"`
	Messages           []Message `json:"messages"`
	System             string    `json:"system,omitempty"`
	MaxTokens          int32     `json:"maxTokens,omitempty"`
	Temperature        float32   `json:"temperature,omitempty"`
	ExecutionRunID     string    `json:"executionRunId"`
	ExecutionBudgetUsd float64   `json:"executionBudgetUsd,omitempty"`
}

// Message represents a conversation message with one or more content blocks.
type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// ContentBlock represents a single content block within a message.
type ContentBlock struct {
	// Type is the block type: "text", "efs_document", "image", or "document".
	Type string `json:"type"`

	// Text content (for type "text").
	Text string `json:"text,omitempty"`

	// EFS file path (for type "efs_document"). Relative to compute node data dir.
	Path string `json:"path,omitempty"`

	// Format hint (for type "efs_document", "image", "document").
	Format string `json:"format,omitempty"`

	// Base64-encoded data (for type "image" or "document").
	Data string `json:"data,omitempty"`

	// Media type (for type "image" or "document").
	MediaType string `json:"mediaType,omitempty"`

	// Document name (for type "document").
	Name string `json:"name,omitempty"`
}

// InvokeResponse is the response from a successful LLM invocation.
type InvokeResponse struct {
	Content         []ResponseContent `json:"content"`
	Model           string            `json:"model"`
	Usage           UsageInfo         `json:"usage"`
	BudgetRemaining BudgetInfo        `json:"budgetRemaining"`
	StopReason      string            `json:"stopReason,omitempty"`
}

// ResponseContent represents a content block in the model's response.
type ResponseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// UsageInfo holds token usage and cost information.
type UsageInfo struct {
	InputTokens      int64   `json:"inputTokens"`
	OutputTokens     int64   `json:"outputTokens"`
	EstimatedCostUsd float64 `json:"estimatedCostUsd"`
}

// BudgetInfo holds remaining budget information.
type BudgetInfo struct {
	BudgetPeriod          string  `json:"budgetPeriod"`
	PeriodBudgetUsd       float64 `json:"periodBudgetUsd"`
	PeriodUsedUsd         float64 `json:"periodUsedUsd"`
	PeriodRemainingUsd    float64 `json:"periodRemainingUsd"`
	ExecutionBudgetUsd    float64 `json:"executionBudgetUsd,omitempty"`
	ExecutionUsedUsd      float64 `json:"executionUsedUsd,omitempty"`
	ExecutionRemainingUsd float64 `json:"executionRemainingUsd,omitempty"`
}

// CheckBudgetResponse is the response from a check-budget action.
type CheckBudgetResponse struct {
	BudgetPeriod          string  `json:"budgetPeriod"`
	PeriodBudgetUsd       float64 `json:"periodBudgetUsd"`
	PeriodUsedUsd         float64 `json:"periodUsedUsd"`
	PeriodRemainingUsd    float64 `json:"periodRemainingUsd"`
	ExecutionBudgetUsd    float64 `json:"executionBudgetUsd,omitempty"`
	ExecutionUsedUsd      float64 `json:"executionUsedUsd,omitempty"`
	ExecutionRemainingUsd float64 `json:"executionRemainingUsd,omitempty"`
}

// ModelInfo represents a model in the list-models response.
type ModelInfo struct {
	ModelID string `json:"modelId"`
	Status  string `json:"status"`
	Hint    string `json:"hint,omitempty"`
}

// ListModelsResponse is the response from a list-models action.
type ListModelsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ErrorResponse is returned by the governor on errors.
type ErrorResponse struct {
	Error           string      `json:"error"`
	Message         string      `json:"message"`
	AllowedModels   []string    `json:"allowedModels,omitempty"`
	BudgetRemaining *BudgetInfo `json:"budgetRemaining,omitempty"`
	MaxSizeBytes    int64       `json:"maxSizeBytes,omitempty"`
	RetryAfterSec   int         `json:"retryAfterSeconds,omitempty"`
	Model           string      `json:"model,omitempty"`
}