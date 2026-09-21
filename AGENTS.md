sbsllm (side by side llm), is an app that lets the user test the same prompt against multiple llm chats (websites) at the same time.

The app starts a very minimal web browser (Chromium via Playwright) to display AI chat websites.
The browser is controlled using Playwright to open the predefined ai chat apps.
The user is prompted to login in those ai chat apps.
The prompt is sent from cli with fanout to all the opened browser pages.
The supported chats are selected using a config file.

the url of kimi is kimi.ai, not kimi.com
The google website the content is inserted in the chat, but not sent
zai chat has the errors as shown in the logs
grok chat is not working at all, not inserting the inputs in the chat.

There is also an design problem. To support openai compatible servers we have to do some heavy lifting, stripping all the chat interface from the inputswe send to the web chats. Over to the chats we only want to send the actual message insert in the local chat (like open-webui). Then we should copy backthe response and send it back to the local chat, instead of the reply "Prompt sent to grok successfully.". Ideally with thinking support (if the webchats shows the thinking trace) and streaming support.

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
