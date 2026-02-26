# pennsieve-llm

Go SDK for invoking LLM models from Pennsieve compute node processors.

This package wraps the LLM Governor Lambda, handling serialization, error parsing, and providing a clean API for common patterns. It works with both ECS and Lambda processors.

## Install

```bash
go get github.com/pennsieve/pennsieve-llm
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/pennsieve/pennsieve-llm/llm"
)

func main() {
    ctx := context.Background()
    gov := llm.NewGovernor()

    // Check if LLM access is available on this compute node
    if !gov.Available() {
        log.Println("LLM access not enabled on this compute node")
        return
    }

    answer, err := gov.Ask(ctx, llm.ModelHaiku45, "What is mitosis?")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(answer)
}
```

The SDK reads two environment variables automatically:

| Variable | Set by | Description |
|----------|--------|-------------|
| `LLM_GOVERNOR_FUNCTION` | Platform (ASL converter) | Governor Lambda function name |
| `EXECUTION_RUN_ID` | Platform (Step Functions) | Current workflow execution ID |

Both are injected by the platform — processor authors don't need to set them.

## Usage

### Simple text prompt

```go
gov := llm.NewGovernor()

answer, err := gov.Ask(ctx, llm.ModelHaiku45, "Explain PCR in one paragraph.")
```

### Text prompt with system instruction

```go
answer, err := gov.AskWithSystem(ctx, llm.ModelSonnet46,
    "You are a biomedical NLP assistant. Return JSON only.",
    "Extract all gene names from this abstract: ...",
)
```

### Ask about a file on EFS

```go
answer, err := gov.AskAboutFile(ctx, llm.ModelSonnet46,
    "Summarize the key findings in this paper.",
    "workdir/run-1/output/paper.pdf",
)
```

The governor reads the file from EFS, detects the format from the extension, and converts it to the appropriate Bedrock content block. Supported formats: PDF, CSV, TXT, MD, HTML, DOC, DOCX, XLS, XLSX, PNG, JPEG, GIF, WEBP.

### Full control with InvokeRequest

```go
resp, err := gov.Invoke(ctx, &llm.InvokeRequest{
    Model:       llm.ModelHaiku45,
    System:      "Extract diagnosis codes as a JSON array.",
    MaxTokens:   2048,
    Temperature: 0.0,
    Messages: []llm.Message{
        llm.UserMessage(
            llm.TextBlock("Extract ICD-10 codes from this clinical note:"),
            llm.FileBlock("input/run-1/src-1/note.txt"),
        ),
    },
    ExecutionBudgetUsd: 1.00,
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(resp.Text())
fmt.Printf("Cost: $%.4f\n", resp.Usage.EstimatedCostUsd)
fmt.Printf("Budget remaining: $%.2f\n", resp.BudgetRemaining.PeriodRemainingUsd)
```

### Multi-turn conversation

```go
resp, err := gov.Invoke(ctx, &llm.InvokeRequest{
    Model: llm.ModelHaiku45,
    Messages: []llm.Message{
        llm.UserMessage(llm.TextBlock("What is the capital of France?")),
        llm.AssistantMessage(llm.TextBlock("The capital of France is Paris.")),
        llm.UserMessage(llm.TextBlock("What is its population?")),
    },
})
```

### Check budget

```go
budget, err := gov.CheckBudget(ctx)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Period: %s, Used: $%.4f, Remaining: $%.2f\n",
    budget.BudgetPeriod, budget.PeriodUsedUsd, budget.PeriodRemainingUsd)
```

### List available models

```go
models, err := gov.ListModels(ctx)
if err != nil {
    log.Fatal(err)
}
for _, m := range models.Models {
    fmt.Printf("%s: %s\n", m.ModelID, m.Status)
}
```

## Error Handling

The SDK returns typed errors for governor-specific failures:

```go
resp, err := gov.Ask(ctx, llm.ModelHaiku45, "Hello")
if err != nil {
    if ge, ok := llm.IsGovernorError(err); ok {
        switch {
        case ge.IsBudgetExceeded():
            fmt.Printf("Budget exhausted. Remaining: $%.2f\n",
                ge.BudgetRemaining.PeriodRemainingUsd)
        case ge.IsModelNotAllowed():
            fmt.Printf("Model not allowed. Available: %v\n", ge.AllowedModels)
        case ge.IsProviderNotAllowed():
            fmt.Println("Provider not approved for this deployment mode")
        case ge.IsModelNotEnabled():
            fmt.Println("Enable the model in the AWS Bedrock console")
        case ge.IsThrottled():
            fmt.Printf("Rate limited. Retry after %d seconds\n", ge.RetryAfterSec)
        }
    }
    log.Fatal(err)
}
```

## Available Models

| Constant | Model ID | Best for |
|----------|----------|----------|
| `llm.ModelHaiku45` | `anthropic.claude-haiku-4-5-20251001` | Fast, low-cost tasks: classification, extraction, simple Q&A |
| `llm.ModelSonnet46` | `anthropic.claude-sonnet-4-6-20250514` | Complex reasoning, analysis, summarization |
| `llm.ModelSonnet4` | `anthropic.claude-sonnet-4-20250514` | Same as Sonnet 4.6 (alias) |

## Content Block Builders

| Builder | Type | Description |
|---------|------|-------------|
| `llm.TextBlock(text)` | `text` | Plain text content |
| `llm.FileBlock(path)` | `efs_document` | EFS file (auto-detected format) |
| `llm.ImageBlock(format, data)` | `image` | Base64-encoded image |
| `llm.UserMessage(blocks...)` | — | User message from content blocks |
| `llm.AssistantMessage(blocks...)` | — | Assistant message from content blocks |

## Configuration Options

```go
// Override function name (default: LLM_GOVERNOR_FUNCTION env var)
gov := llm.NewGovernor(llm.WithFunctionName("custom-governor"))

// Override execution run ID (default: EXECUTION_RUN_ID env var)
gov := llm.NewGovernor(llm.WithExecutionRunID("my-run-id"))

// Provide a custom Lambda client (useful for testing)
gov := llm.NewGovernor(llm.WithLambdaClient(myClient))
```