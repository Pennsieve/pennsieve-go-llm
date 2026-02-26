package llm

import "fmt"

// GovernorError represents an error returned by the LLM Governor.
type GovernorError struct {
	Code            string
	Msg             string
	AllowedModels   []string
	BudgetRemaining *BudgetInfo
	RetryAfterSec   int
}

func (e *GovernorError) Error() string {
	return fmt.Sprintf("governor error [%s]: %s", e.Code, e.Msg)
}

// IsBudgetExceeded returns true if the error is a budget_exceeded error.
func (e *GovernorError) IsBudgetExceeded() bool {
	return e.Code == "budget_exceeded"
}

// IsModelNotAllowed returns true if the error is a model_not_allowed error.
func (e *GovernorError) IsModelNotAllowed() bool {
	return e.Code == "model_not_allowed"
}

// IsProviderNotAllowed returns true if the error is a provider_not_allowed error.
func (e *GovernorError) IsProviderNotAllowed() bool {
	return e.Code == "provider_not_allowed"
}

// IsModelNotEnabled returns true if the model needs to be enabled in Bedrock console.
func (e *GovernorError) IsModelNotEnabled() bool {
	return e.Code == "model_not_enabled"
}

// IsThrottled returns true if the request was rate-limited.
func (e *GovernorError) IsThrottled() bool {
	return e.Code == "bedrock_throttled"
}

// IsGovernorError checks whether an error is a GovernorError and returns it.
func IsGovernorError(err error) (*GovernorError, bool) {
	if ge, ok := err.(*GovernorError); ok {
		return ge, true
	}
	return nil, false
}