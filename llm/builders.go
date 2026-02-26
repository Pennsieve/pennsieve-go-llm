package llm

// TextBlock creates a text content block.
func TextBlock(text string) ContentBlock {
	return ContentBlock{Type: "text", Text: text}
}

// FileBlock creates an efs_document content block from a file path.
// The path is relative to the compute node's data directory on EFS.
func FileBlock(path string) ContentBlock {
	return ContentBlock{Type: "efs_document", Path: path}
}

// ImageBlock creates an inline image content block from base64-encoded data.
func ImageBlock(format, base64Data string) ContentBlock {
	return ContentBlock{Type: "image", Format: format, Data: base64Data}
}

// UserMessage creates a user message with the given content blocks.
func UserMessage(blocks ...ContentBlock) Message {
	return Message{Role: "user", Content: blocks}
}

// AssistantMessage creates an assistant message with the given content blocks.
func AssistantMessage(blocks ...ContentBlock) Message {
	return Message{Role: "assistant", Content: blocks}
}