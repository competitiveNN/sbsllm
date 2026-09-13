sbsllm (side by side llm), is an app that lets the user test the same prompt against multiple llm chats (websites) at the same time.

The app starts a very minimal web browser (Chromium via Playwright) to display AI chat websites.
The browser is controlled using Playwright to open the predefined ai chat apps.
The user is prompted to login in those ai chat apps.
The prompt is sent from cli with fanout to all the opened browser pages.
The supported chats are selected using a config file.

The apps supports:
# chatgpt
https://chatgpt.com/
# qwen
https://chat.qwen.ai/
# deepseek
https://chat.deepseek.com
# claude
https://claude.ai/
# grok
https://grok.com/
# google
https://aistudio.google.com/
# mistral
https://chat.mistral.ai
# kimi
https://www.kimi.com/
# zai
https://chat.z.ai/auth
# meta
https://meta.ai/
# huggingface
https://huggingface.co/chat
# tencent
https://aistudio.tencent.ai/
