mkdir -p ~/.claude
rm -f ~/.claude/settings.json
cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-5-20251001",
    "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-5-20250929",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5-20251001",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-5-20251101"
  }
}
EOF


export ANTHROPIC_BASE_URL="https://api.gemai.cc"
export ANTHROPIC_AUTH_TOKEN="sk-5MHhBrORA7q8xrqvUCYUzS31WrzBCq1pa5jWJiHu9L2G4CAg"

IS_SANDBOX=1 claude --continue --dangerously-skip-permissions