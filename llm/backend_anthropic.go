package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// modelMap maps Bedrock inference profile IDs to Anthropic API model names.
var modelMap = map[string]string{
	ModelHaiku45:  "claude-haiku-4-5-20251001",
	ModelSonnet45: "claude-sonnet-4-5-20250929",
	ModelSonnet46: "claude-sonnet-4-6",
	ModelSonnet4:  "claude-sonnet-4-20250514",
}

// MapModel converts a Bedrock inference profile ID to an Anthropic API model name.
func MapModel(bedrockID string) string {
	if name, ok := modelMap[bedrockID]; ok {
		return name
	}
	if strings.HasPrefix(bedrockID, "us.anthropic.") {
		return strings.TrimPrefix(bedrockID, "us.anthropic.")
	}
	return bedrockID
}

// docMediaTypes maps file extensions to MIME types.
var docMediaTypes = map[string]string{
	"pdf":  "application/pdf",
	"csv":  "text/csv",
	"txt":  "text/plain",
	"md":   "text/markdown",
	"html": "text/html",
}

func docMediaType(ext string) string {
	if mt, ok := docMediaTypes[ext]; ok {
		return mt
	}
	return "application/octet-stream"
}

// AnthropicBackend calls the Anthropic Messages API directly using the
// user's API key. This is intended for local development.
type AnthropicBackend struct {
	apiKey     string
	httpClient *http.Client
}

// AnthropicOption configures an AnthropicBackend.
type AnthropicOption func(*AnthropicBackend)

// WithAPIKey sets the API key. By default it is read from ANTHROPIC_API_KEY.
func WithAPIKey(key string) AnthropicOption {
	return func(b *AnthropicBackend) {
		b.apiKey = key
	}
}

// WithHTTPClient provides a custom HTTP client (useful for testing).
func WithHTTPClient(client *http.Client) AnthropicOption {
	return func(b *AnthropicBackend) {
		b.httpClient = client
	}
}

// NewAnthropicBackend creates a new Anthropic backend.
func NewAnthropicBackend(opts ...AnthropicOption) *AnthropicBackend {
	b := &AnthropicBackend{
		apiKey:     os.Getenv("ANTHROPIC_API_KEY"),
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// Anthropic Messages API request/response types.

type anthropicRequest struct {
	Model       string                   `json:"model"`
	MaxTokens   int32                    `json:"max_tokens"`
	System      string                   `json:"system,omitempty"`
	Temperature float32                  `json:"temperature,omitempty"`
	Messages    []map[string]interface{} `json:"messages"`
}

type anthropicResponse struct {
	Content    []anthropicContentBlock `json:"content"`
	Model      string                 `json:"model"`
	StopReason string                 `json:"stop_reason"`
	Usage      struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func (b *AnthropicBackend) Invoke(ctx context.Context, req *InvokeRequest) (*InvokeResponse, error) {
	model := MapModel(req.Model)

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	apiReq := anthropicRequest{
		Model:       model,
		MaxTokens:   maxTokens,
		System:      req.System,
		Temperature: req.Temperature,
		Messages:    convertMessages(req.Messages),
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Anthropic request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", b.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Anthropic API request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp anthropicErrorResponse
		if json.Unmarshal(respBody, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Anthropic API error (%s): %s", errResp.Error.Type, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Anthropic API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Anthropic response: %w", err)
	}

	content := make([]ResponseContent, 0, len(apiResp.Content))
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			content = append(content, ResponseContent{Type: "text", Text: block.Text})
		}
	}

	return &InvokeResponse{
		Content: content,
		Model:   apiResp.Model,
		Usage: UsageInfo{
			InputTokens:  apiResp.Usage.InputTokens,
			OutputTokens: apiResp.Usage.OutputTokens,
		},
		StopReason: apiResp.StopReason,
	}, nil
}

func (b *AnthropicBackend) CheckBudget(_ context.Context, _ string) (*CheckBudgetResponse, error) {
	return &CheckBudgetResponse{
		BudgetPeriod:          "none",
		PeriodRemainingUsd:    math.Inf(1),
		ExecutionRemainingUsd: math.Inf(1),
	}, nil
}

func (b *AnthropicBackend) ListModels(_ context.Context) (*ListModelsResponse, error) {
	return &ListModelsResponse{Models: allModels()}, nil
}

// convertMessages converts SDK messages to Anthropic API format.
func convertMessages(messages []Message) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(messages))
	for _, msg := range messages {
		blocks := make([]interface{}, 0, len(msg.Content))
		for _, block := range msg.Content {
			blocks = append(blocks, convertContentBlock(block))
		}
		result = append(result, map[string]interface{}{
			"role":    msg.Role,
			"content": blocks,
		})
	}
	return result
}

func convertContentBlock(block ContentBlock) map[string]interface{} {
	switch block.Type {
	case "text":
		return map[string]interface{}{
			"type": "text",
			"text": block.Text,
		}
	case "image":
		mediaType := block.MediaType
		if mediaType == "" && block.Format != "" {
			mediaType = "image/" + block.Format
		}
		return map[string]interface{}{
			"type": "image",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": mediaType,
				"data":       block.Data,
			},
		}
	case "document":
		mediaType := block.MediaType
		if mediaType == "" {
			mediaType = docMediaType(block.Format)
		}
		return map[string]interface{}{
			"type": "document",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": mediaType,
				"data":       block.Data,
			},
		}
	case "efs_document":
		return convertEFSDocument(block)
	default:
		return map[string]interface{}{
			"type": "text",
			"text": fmt.Sprintf("[Unsupported content block: %s]", block.Type),
		}
	}
}

func convertEFSDocument(block ContentBlock) map[string]interface{} {
	if block.Path == "" {
		return map[string]interface{}{
			"type": "text",
			"text": "[File not available locally: empty path]",
		}
	}

	data, err := os.ReadFile(block.Path)
	if err != nil {
		return map[string]interface{}{
			"type": "text",
			"text": fmt.Sprintf("[File not available locally: %s]", block.Path),
		}
	}

	ext := strings.TrimPrefix(filepath.Ext(block.Path), ".")
	return map[string]interface{}{
		"type": "document",
		"source": map[string]interface{}{
			"type":       "base64",
			"media_type": docMediaType(ext),
			"data":       base64.StdEncoding.EncodeToString(data),
		},
	}
}