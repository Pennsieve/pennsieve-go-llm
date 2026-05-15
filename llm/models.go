package llm

// Well-known Bedrock inference profile IDs for convenience.
// Use "us." prefix for US region on-demand inference profiles (HIPAA-friendly
// default — keeps inference in US AWS regions).
const (
	ModelHaiku45  = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
	ModelSonnet4  = "us.anthropic.claude-sonnet-4-20250514-v1:0"
	ModelSonnet45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
	ModelSonnet46 = "us.anthropic.claude-sonnet-4-6"
	ModelOpus47   = "us.anthropic.claude-opus-4-7"
)