<template>
  <div class="app">
    <aside class="sidebar">
      <h2>会话</h2>
      <div class="session-list">
        <button
          v-for="sid in sessions"
          :key="sid"
          :class="['session', sid === sessionId ? 'active' : '']"
          @click="selectSession(sid)"
        >{{ sid }}</button>
      </div>
      <button class="new-session" @click="newSession">新建会话</button>
      <div class="health">
        <span>后端健康：{{ healthStatus }}</span>
        <button @click="checkHealth">刷新</button>
      </div>
    </aside>
    <main class="chat">
      <h2>智能客服</h2>
      <div class="messages">
        <div
          v-for="(m, i) in messages"
          :key="i"
          :class="['msg', m.role]"
        >
          <strong>{{ m.role === 'user' ? '我' : '客服' }}：</strong>
          <span>{{ m.content }}</span>
        </div>
      </div>
      <div class="input-row">
        <input
          v-model="input"
          placeholder="请输入你的问题..."
          @keyup.enter="send"
        />
        <button @click="send" :disabled="loading">发送</button>
      </div>
    </main>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'

type Msg = { role: 'user' | 'assistant', content: string }

const input = ref('')
const messages = ref<Msg[]>([])
const sessionId = ref('s1')
const sessions = ref<string[]>(['s1'])
const healthStatus = ref('unknown')
const loading = ref(false)

function selectSession(sid: string) {
  sessionId.value = sid
  messages.value = []
}

function newSession() {
  const sid = `s${Date.now()}`
  sessions.value.push(sid)
  selectSession(sid)
}

async function checkHealth() {
  try {
    const res = await fetch('/health')
    healthStatus.value = res.ok ? 'ok' : 'bad'
  } catch {
    healthStatus.value = 'bad'
  }
}

async function send() {
  const text = input.value.trim()
  if (!text) return
  input.value = ''
  messages.value.push({ role: 'user', content: text })
  loading.value = true
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId.value, message: text })
    })
    if (!res.ok) throw new Error('request failed')
    const data = await res.json()
    messages.value.push({ role: 'assistant', content: data.reply })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: '请求失败，请稍后重试' })
  } finally {
    loading.value = false
  }
}

onMounted(checkHealth)
</script>

<style scoped>
.app {
  display: flex;
  height: 100vh;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}
.sidebar {
  width: 280px;
  border-right: 1px solid #e5e5e5;
  padding: 12px;
}
.session-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 8px 0 16px;
}
.session {
  padding: 8px 10px;
  border: 1px solid #ddd;
  background: #fafafa;
  cursor: pointer;
}
.session.active {
  background: #e6f4ff;
  border-color: #6ab4ff;
}
.new-session {
  margin-bottom: 16px;
}
.health {
  display: flex;
  gap: 8px;
  align-items: center;
}
.chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 12px;
}
.messages {
  flex: 1;
  overflow: auto;
  padding: 10px;
  background: #fff;
  border: 1px solid #eee;
  border-radius: 6px;
}
.msg {
  margin: 8px 0;
}
.msg.user {
  text-align: left;
}
.msg.assistant {
  text-align: left;
}
.input-row {
  margin-top: 12px;
  display: flex;
  gap: 8px;
}
input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
}
button {
  padding: 10px 16px;
}
</style>
