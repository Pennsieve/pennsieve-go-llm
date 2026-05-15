# pennsieve-go-llm

Thin Go configuration helper for the Pennsieve LLM Governor.

Returns a pre-configured `*anthropic.Client` (from [anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go)) pointed at the Pennsieve LLM Governor with SigV4 auth and the `x-execution-run-id` header wired up. Streaming, tool use, prompt caching — every Anthropic SDK feature works because you're using the real Anthropic SDK.

## Installation

```bash
go get github.com/pennsieve/pennsieve-go-llm
```

Requires Go 1.24+.

## Quick start

```go
import (
    "context"
    "fmt"

    "github.com/anthropics/anthropic-sdk-go"
    "github.com/pennsieve/pennsieve-go-llm/llm"
)

func main() {
    ctx := context.Background()
    gov, err := llm.New(ctx)  // auto-configures from $LLM_GOVERNOR_URL + AWS creds
    if err != nil {
        panic(err)
    }

    resp, err := gov.Client().Messages.New(ctx, anthropic.MessageNewParams{
        Model:     anthropic.F(llm.ModelSonnet45),
        MaxTokens: anthropic.F(int64(1024)),
        Messages: anthropic.F([]anthropic.MessageParam{
            anthropic.NewUserMessage(anthropic.NewTextBlock("Hello, world!")),
        }),
    })
    if err != nil { panic(err) }
    fmt.Println(resp.Content[0].Text)
}
```

The object returned by `gov.Client()` **is** `*anthropic.Client`. Everything in the [anthropic-sdk-go docs](https://github.com/anthropics/anthropic-sdk-go) applies.

## Configuration

| Env var | Purpose |
|---|---|
| `LLM_GOVERNOR_URL` | Governor Function URL (platform-injected) |
| `EXECUTION_RUN_ID` | Cost attribution; attached as `x-execution-run-id` header (platform-injected) |
| `AWS_REGION` | SigV4 signing region (default `us-east-1`) |

Options:

```go
gov, err := llm.New(ctx,
    llm.WithURL("https://abc.lambda-url.us-east-1.on.aws"),
    llm.WithExecutionRunID("run-123"),
    llm.WithRegion("us-east-1"),
)
```

## Governor-specific endpoints

`CheckBudget` and `ListModels` query Pennsieve-specific endpoints (not part of the Anthropic API):

```go
b, _ := gov.CheckBudget(ctx)
fmt.Printf("$%.2f remaining this %s\n", b.PeriodRemainingUsd, b.BudgetPeriod)

models, _ := gov.ListModels(ctx)
for _, m := range models.Models { fmt.Println(m.ModelID, m.Status) }
```

## Testing

Use `httptest.NewServer` + `llm.WithHTTPClient(ts.Client())` to bypass SigV4 against a fake governor.

For chat (`Messages.New`) tests, mock at the http.Transport level — see the anthropic-sdk-go testing docs.

## Model ID constants

| Constant | Bedrock inference profile ID |
|---|---|
| `ModelHaiku45` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `ModelSonnet4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| `ModelSonnet45` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `ModelSonnet46` | `us.anthropic.claude-sonnet-4-6` |
| `ModelOpus47` | `us.anthropic.claude-opus-4-7` |

`us.*` keeps inference in US AWS regions — HIPAA-friendly default.

## Migration from v0.x

The SDK pivoted to "thin configuration wrapper" — it returns the real `*anthropic.Client` rather than wrapping it with parallel types. The previous `Governor.Ask`, `Governor.Invoke`, and `Governor.AskAboutFile` convenience methods are gone. Use `gov.Client().Messages.New(...)` directly. The migration cost is offset by gaining direct access to streaming, tool use, prompt caching, and every future Anthropic SDK feature.

## License

MIT
