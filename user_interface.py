"""
FZA System — Web Interface
ChatGPT-style: conversation list + persistent memory.
"""
import contextlib
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from main_fza_system import FZAManager


# ── 세션 관리 ────────────────────────────────────────────────
SESSIONS: dict[str, FZAManager] = {}

def get_manager(user_id: str) -> FZAManager:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = FZAManager(user_id=user_id)
    return SESSIONS[user_id]


# ── Web UI ───────────────────────────────────────────────────
WEB_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="theme-color" content="#0a0a12"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
<link rel="manifest" href="/manifest.json"/>
<title>FZA — 망각 없는 AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="window._katexReady=true"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0d0d;--surface:#141414;--card:#1c1c1c;
  --border:rgba(255,255,255,.07);--border2:rgba(255,255,255,.1);
  --rose:#f43f5e;--rose-dim:#e11d48;--rose-glow:rgba(244,63,94,.15);
  --green:#22c55e;--amber:#f59e0b;--blue:#38bdf8;
  --text:#f1f1f1;--text2:#9a9a9a;--text3:#555;
  --r:16px;--r-sm:10px;--r-pill:100px;--sidebar:260px;
}
html,body{height:100%;overflow:hidden;font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text)}

/* ── LAYOUT ── */
.app{display:flex;height:100vh}

/* ── SIDEBAR ── */
.sidebar{
  width:var(--sidebar);flex-shrink:0;
  background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}
.sb-top{padding:16px 14px 12px;flex-shrink:0}
.logo{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.logo-icon{
  width:32px;height:32px;border-radius:10px;flex-shrink:0;
  background:var(--rose);display:flex;align-items:center;justify-content:center;
  font-size:16px;box-shadow:0 4px 12px var(--rose-glow);
}
.logo-text{font-size:14px;font-weight:700;letter-spacing:-.3px}
.logo-sub{font-size:9px;color:var(--text3);letter-spacing:.6px;text-transform:uppercase;margin-top:1px}

.new-chat-btn{
  display:flex;align-items:center;justify-content:center;gap:7px;
  width:100%;padding:10px;border-radius:var(--r-pill);
  background:var(--rose);color:#fff;
  border:none;font-size:13px;font-weight:600;cursor:pointer;
  transition:all .2s;box-shadow:0 4px 14px var(--rose-glow);
}
.new-chat-btn:hover{background:var(--rose-dim);transform:translateY(-1px);box-shadow:0 6px 18px var(--rose-glow)}
.new-chat-btn:active{transform:none}

/* conversation list */
.conv-list{flex:1;overflow-y:auto;padding:6px 8px 0}
.conv-list::-webkit-scrollbar{width:2px}
.conv-list::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

.conv-group-title{
  font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
  color:var(--text3);padding:10px 8px 4px;
}
.conv-item{
  display:flex;align-items:center;justify-content:space-between;
  padding:9px 12px;border-radius:var(--r-sm);cursor:pointer;
  transition:background .15s;margin-bottom:1px;
  font-size:13px;color:var(--text2);
}
.conv-item:hover{background:var(--card)}
.conv-item.active{background:var(--card);color:var(--text)}
.conv-title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.conv-turns{font-size:10px;color:var(--text3);flex-shrink:0;margin-left:6px}

/* memory panel (bottom) */
.sb-mem{border-top:1px solid var(--border);flex-shrink:0}
.sb-mem-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:11px 16px;cursor:pointer;
  font-size:11px;font-weight:600;color:var(--text2);
  user-select:none;
}
.sb-mem-header:hover{color:var(--text)}
.mem-toggle{font-size:10px;color:var(--text3);transition:transform .2s}
.mem-toggle.open{transform:rotate(180deg)}

.sb-mem-body{padding:0 12px 12px;overflow-y:auto;max-height:200px;display:none}
.sb-mem-body.open{display:block}
.sb-mem-body::-webkit-scrollbar{width:2px}
.sb-mem-body::-webkit-scrollbar-thumb{background:var(--border2)}

.stats-row{display:flex;gap:6px;margin-bottom:8px}
.stat-pill{flex:1;text-align:center;padding:8px 4px;border-radius:var(--r-sm);background:var(--card)}
.stat-num{font-size:15px;font-weight:700}
.stat-num.c{color:var(--rose)}.stat-num.g{color:var(--green)}
.stat-lbl{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px}

.mem-items{display:flex;flex-direction:column;gap:3px}
.mem-item{padding:6px 10px;border-radius:8px;background:var(--card);font-size:11px;color:var(--text2);line-height:1.4}
.mem-item.new{background:rgba(244,63,94,.08);color:var(--text);animation:pop .4s ease}
@keyframes pop{from{transform:scale(.97);opacity:.5}to{transform:scale(1);opacity:1}}
.empty-hint{font-size:11px;color:var(--text3);font-style:italic;padding:2px 4px}

/* user chip */
.user-chip{
  display:flex;align-items:center;gap:8px;padding:8px 10px;
  border-radius:var(--r-sm);cursor:pointer;transition:background .15s;margin-top:4px;
}
.user-chip:hover{background:var(--card)}
.avatar{
  width:28px;height:28px;border-radius:50%;flex-shrink:0;
  background:var(--rose);display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;color:#fff;
}
.user-name{font-size:12px;font-weight:600}
.user-sub{font-size:10px;color:var(--text3)}

/* ── MAIN ── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;height:52px;
  background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0;
}
.chat-title{font-size:13px;font-weight:600;color:var(--text2);max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pills{display:flex;gap:6px}
.pill{
  display:flex;align-items:center;gap:4px;padding:4px 10px;border-radius:var(--r-pill);
  font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.3px;
  background:var(--card);transition:all .25s;
}
.pill .dot{width:5px;height:5px;border-radius:50%;background:var(--text3)}
.pill.on .dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.pill.on{color:var(--green)}

#messages{
  flex:1;overflow-y:auto;padding:24px 28px;
  display:flex;flex-direction:column;gap:16px;
}
#messages::-webkit-scrollbar{width:3px}
#messages::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

.msg{display:flex;flex-direction:column;max-width:72%;animation:fadeUp .2s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.bot{align-self:flex-start;align-items:flex-start}
.msg.sys{align-self:center;align-items:center;max-width:88%}

.bubble{
  padding:12px 16px;border-radius:var(--r);
  font-size:14px;line-height:1.7;white-space:pre-wrap;word-break:break-word;
}
.msg.user .bubble{
  background:var(--rose);color:#fff;border-bottom-right-radius:4px;
  box-shadow:0 4px 16px var(--rose-glow);
}
.msg.bot .bubble{
  background:var(--card);color:var(--text);border-bottom-left-radius:4px;
  box-shadow:0 2px 8px rgba(0,0,0,.3);
}
.msg.sys .bubble{
  background:rgba(34,197,94,.07);color:var(--green);
  border:1px solid rgba(34,197,94,.2);
  font-size:12px;padding:6px 14px;border-radius:var(--r-pill);
}
.msg-meta{font-size:10px;color:var(--text3);margin-top:4px;padding:0 4px}

/* ── KaTeX ── */
.bubble .katex-display{margin:10px 0;overflow-x:auto;overflow-y:hidden;padding:4px 0}
.bubble .katex{font-size:1.05em}
.msg.user .bubble .katex,.msg.user .bubble .katex *{color:#fff}
.msg.bot  .bubble .katex,.msg.bot  .bubble .katex *{color:var(--text)}

.mem-toast{
  display:flex;align-items:center;gap:5px;margin-top:6px;
  padding:4px 12px;background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.2);
  border-radius:var(--r-pill);font-size:11px;color:var(--rose);animation:fadeUp .3s ease;
}

.typing{display:flex;gap:4px;align-items:center;padding:12px 16px}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--text3);animation:blink 1.2s infinite}
.typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}

/* ── INPUT ── */
.input-area{padding:12px 20px 16px;background:var(--surface);border-top:1px solid var(--border);flex-shrink:0}
.input-hint{font-size:11px;color:var(--text3);margin-bottom:8px;display:flex;gap:12px;flex-wrap:wrap}
.input-hint span{cursor:pointer;transition:color .2s}.input-hint span:hover{color:var(--rose)}
.composer{
  display:flex;align-items:flex-end;
  background:var(--card);border-radius:var(--r);
  padding:6px 6px 6px 14px;
  box-shadow:0 2px 12px rgba(0,0,0,.3);
}
#msg{
  flex:1;background:transparent;border:none;
  padding:8px 4px;font-size:14px;color:var(--text);font-family:'Inter',sans-serif;
  outline:none;resize:none;max-height:120px;min-height:36px;line-height:1.5;
}
#msg::placeholder{color:var(--text3)}
.attach-btn,.voice-btn{
  background:none;border:none;cursor:pointer;font-size:16px;
  padding:8px 9px;border-radius:8px;color:var(--text3);
  transition:all .2s;flex-shrink:0;line-height:1;
}
.attach-btn:hover,.voice-btn:hover{background:var(--surface);color:var(--rose)}
.voice-btn.listening{color:#ef4444!important;animation:rec-pulse 1.2s infinite}
@keyframes rec-pulse{0%,100%{opacity:1}50%{opacity:.35}}
#send{
  padding:9px 20px;border-radius:var(--r-sm);background:var(--rose);color:#fff;
  border:none;font-size:13px;font-weight:600;cursor:pointer;
  transition:all .2s;flex-shrink:0;margin-left:4px;
  box-shadow:0 2px 8px var(--rose-glow);
}
#send:hover{background:var(--rose-dim);transform:translateY(-1px)}
#send:disabled{opacity:.4;cursor:default;transform:none}

/* ── MODAL ── */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:100;backdrop-filter:blur(6px)}
.modal{background:var(--surface);border:1px solid var(--border2);border-radius:20px;padding:32px;width:360px;box-shadow:0 30px 70px rgba(0,0,0,.6)}
.modal h2{font-size:20px;font-weight:700;margin-bottom:8px;letter-spacing:-.3px}
.modal p{font-size:13px;color:var(--text2);line-height:1.6;margin-bottom:20px}
.modal input{
  width:100%;background:var(--card);border:1px solid var(--border2);
  border-radius:var(--r-sm);padding:12px 14px;font-size:14px;color:var(--text);
  font-family:'Inter',sans-serif;outline:none;margin-bottom:12px;transition:border-color .2s;
}
.modal input:focus{border-color:var(--rose)}
.modal button{
  width:100%;padding:13px;background:var(--rose);color:#fff;
  border:none;border-radius:var(--r-pill);font-size:14px;font-weight:600;
  cursor:pointer;transition:all .2s;box-shadow:0 4px 12px var(--rose-glow);
}
.modal button:hover{background:var(--rose-dim)}

/* ── MEMORY EXPLORER ── */
.me-modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;align-items:center;justify-content:center;z-index:200;backdrop-filter:blur(8px)}
.me-modal-bg.open{display:flex}
.me-modal{background:var(--surface);border:1px solid var(--border2);border-radius:20px;width:580px;max-width:96vw;max-height:82vh;display:flex;flex-direction:column;box-shadow:0 30px 80px rgba(0,0,0,.7)}
.me-head{display:flex;align-items:center;justify-content:space-between;padding:18px 22px 14px;border-bottom:1px solid var(--border);flex-shrink:0}
.me-head h3{font-size:15px;font-weight:700;letter-spacing:-.2px}
.me-close{background:none;border:none;color:var(--text3);font-size:18px;cursor:pointer;padding:4px 8px;border-radius:8px;transition:all .2s}
.me-close:hover{color:var(--text);background:var(--card)}
.me-search{padding:12px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.me-search input{width:100%;background:var(--card);border:none;border-radius:var(--r-sm);padding:9px 14px;font-size:13px;color:var(--text);font-family:'Inter',sans-serif;outline:none}
.me-search input:focus{box-shadow:0 0 0 2px var(--rose)}
.me-tabs{display:flex;padding:0 16px;border-bottom:1px solid var(--border);flex-shrink:0;gap:4px}
.me-tab{padding:10px 14px;font-size:12px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent;color:var(--text3);transition:all .2s;user-select:none}
.me-tab.active{color:var(--rose);border-color:var(--rose)}
.me-body{flex:1;overflow-y:auto;padding:8px 10px}
.me-body::-webkit-scrollbar{width:2px}
.me-body::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}
.me-count{font-size:11px;color:var(--text3);padding:4px 6px 8px}
.me-item{display:flex;align-items:flex-start;gap:8px;padding:7px 10px;border-radius:10px;transition:background .15s}
.me-item:hover{background:var(--card)}
.me-item-key{font-size:11px;font-weight:600;color:var(--rose);min-width:70px;max-width:90px;flex-shrink:0;padding-top:2px;word-break:break-word}
.me-item-text{flex:1;font-size:12px;color:var(--text2);line-height:1.5;word-break:break-word}
.me-item-btns{display:flex;gap:2px;flex-shrink:0;opacity:0;transition:opacity .15s}
.me-item:hover .me-item-btns{opacity:1}
.me-btn{background:none;border:none;cursor:pointer;padding:3px 7px;border-radius:6px;font-size:12px;transition:background .15s;color:var(--text3)}
.me-btn:hover{background:var(--border2);color:var(--text)}
.me-btn.del:hover{background:rgba(239,68,68,.12);color:#ef4444}
.me-edit-row{padding:4px 10px 8px;display:none}
.me-edit-row.open{display:flex;gap:6px;align-items:center}
.me-edit-input{flex:1;background:var(--card);border:1px solid var(--rose);border-radius:8px;padding:7px 11px;font-size:12px;color:var(--text);font-family:'Inter',sans-serif;outline:none}
.me-edit-save{background:var(--rose);color:#fff;border:none;border-radius:var(--r-pill);padding:7px 14px;font-size:11px;font-weight:600;cursor:pointer;white-space:nowrap}
.me-edit-cancel{background:var(--card);color:var(--text2);border:none;border-radius:var(--r-pill);padding:7px 12px;font-size:11px;cursor:pointer;white-space:nowrap}
.me-add{padding:12px 16px;border-top:1px solid var(--border);flex-shrink:0}
.me-add-row{display:flex;gap:8px}
.me-add-input{flex:1;background:var(--card);border:none;border-radius:var(--r-sm);padding:9px 14px;font-size:13px;color:var(--text);font-family:'Inter',sans-serif;outline:none}
.me-add-input:focus{box-shadow:0 0 0 2px var(--rose)}
.me-add-btn{background:var(--rose);color:#fff;border:none;border-radius:var(--r-pill);padding:9px 18px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;transition:all .2s;box-shadow:0 2px 8px var(--rose-glow)}
.me-add-btn:hover{background:var(--rose-dim)}
.me-empty{text-align:center;padding:28px;font-size:12px;color:var(--text3);font-style:italic}
.sb-mem-edit-btn{background:none;border:none;cursor:pointer;font-size:11px;color:var(--text3);padding:2px 6px;border-radius:6px;transition:color .2s;flex-shrink:0}
.sb-mem-edit-btn:hover{color:var(--rose)}

/* ── 추가 컴포넌트 ── */
.cat-badge{display:inline-flex;align-items:center;font-size:9px;padding:2px 7px;border-radius:var(--r-pill);font-weight:700;flex-shrink:0;letter-spacing:.3px}
.mem-ts{font-size:10px;color:var(--text3);margin-top:2px}
.conv-del-btn{background:none;border:none;cursor:pointer;color:var(--text3);font-size:10px;padding:2px 5px;border-radius:4px;opacity:0;transition:all .15s;flex-shrink:0;line-height:1}
.conv-item:hover .conv-del-btn{opacity:1}
.conv-del-btn:hover{background:rgba(239,68,68,.12)!important;color:#ef4444!important}
.upload-hint{font-size:11px;color:var(--text2);padding:6px 14px;background:var(--card);border-radius:var(--r-sm);margin-bottom:8px;display:none;animation:fadeUp .2s ease}
.upload-hint.visible{display:block}
</style>
</head>
<body>

<!-- MEMORY EXPLORER -->
<div class="me-modal-bg" id="me-modal">
  <div class="me-modal">
    <div class="me-head">
      <h3>🧠 기억 관리</h3>
      <button class="me-close" onclick="closeMemExplorer()">✕</button>
    </div>
    <div class="me-search">
      <input id="me-search-input" placeholder="기억 검색..." oninput="renderMemExplorer()"/>
    </div>
    <div class="me-tabs">
      <div class="me-tab active" id="tab-mems" onclick="switchMemTab('mems')">자동 기억 <span id="tab-mems-count"></span></div>
      <div class="me-tab" id="tab-profile" onclick="switchMemTab('profile')">프로필 <span id="tab-profile-count"></span></div>
    </div>
    <div class="me-body" id="me-body"></div>
    <div class="me-add" id="me-add-section">
      <div class="me-add-row">
        <input class="me-add-input" id="me-add-input" placeholder="새 기억 직접 추가..."/>
        <button class="me-add-btn" onclick="submitAddMemory()">추가</button>
      </div>
    </div>
  </div>
</div>

<!-- ONBOARDING -->
<div class="modal-bg" id="modal">
  <div class="modal">
    <h2>👋 안녕하세요!</h2>
    <p>이름을 알려주시면 대화를 나눌수록 더 잘 기억하게 됩니다.</p>
    <input id="modal-name" placeholder="이름을 입력하세요" maxlength="30" autofocus/>
    <button onclick="startSession()">시작하기</button>
  </div>
</div>

<div class="app" id="app" style="display:none">

  <!-- SIDEBAR -->
  <aside class="sidebar">
    <div class="sb-top">
      <div class="logo">
        <span class="logo-icon">🌿</span>
        <div>
          <div class="logo-text">FZA</div>
          <div class="logo-sub">Forgetting-Zero AI</div>
        </div>
      </div>
      <button class="new-chat-btn" onclick="newChat()">＋ 새 채팅</button>
    </div>

    <div class="conv-list" id="conv-list"></div>

    <!-- Memory panel -->
    <div class="sb-mem">
      <div class="user-chip" onclick="changeUser()">
        <div class="avatar" id="avatar">?</div>
        <div>
          <div class="user-name" id="username-display">—</div>
          <div class="user-sub">계정 변경</div>
        </div>
      </div>
      <div class="sb-mem-header">
        <span onclick="toggleMem()" style="flex:1;cursor:pointer">🧠 기억 현황</span>
        <button class="sb-mem-edit-btn" onclick="openMemExplorer()" title="기억 관리">✏️</button>
        <span class="mem-toggle" id="mem-toggle" onclick="toggleMem()" style="cursor:pointer">▼</span>
      </div>
      <div class="sb-mem-body" id="mem-body">
        <div class="stats-row">
          <div class="stat-pill">
            <div class="stat-num c" id="s-rag">0</div>
            <div class="stat-lbl">RAG</div>
          </div>
          <div class="stat-pill">
            <div class="stat-num g" id="s-mem">0</div>
            <div class="stat-lbl">기억</div>
          </div>
        </div>
        <div class="mem-items" id="mem-list">
          <div class="empty-hint">대화를 나누면 자동으로 기억됩니다</div>
        </div>
      </div>
    </div>
  </aside>

  <!-- MAIN -->
  <div class="main">
    <div class="topbar">
      <span class="chat-title" id="chat-title">새 대화</span>
      <div class="pills">
        <div class="pill" id="pill-rag"><span class="dot"></span>RAG</div>
        <div class="pill on" id="pill-mem"><span class="dot"></span>AUTO-MEM</div>
      </div>
    </div>

    <div id="messages"></div>

    <div class="input-area">
      <div class="input-hint">
        <span onclick="hint('/기억')">📋 /기억</span>
        <span onclick="hint('/수식 추가 ')">📐 /수식 추가</span>
        <span onclick="hint('/초기화')">🗑 /초기화</span>
      </div>
      <div class="upload-hint" id="upload-hint"></div>
      <div class="composer">
        <button class="attach-btn" onclick="triggerFileUpload()" title="파일 첨부 (.txt .pdf .png .jpg)">📎</button>
        <textarea id="msg" placeholder="무엇이든 물어보세요..." rows="1"></textarea>
        <button class="voice-btn" id="voice-btn" onclick="toggleVoice()" title="음성 입력">🎤</button>
        <button id="send">전송</button>
      </div>
      <input type="file" id="file-input" accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.webp" style="display:none" onchange="handleFileUpload(this)"/>
    </div>
  </div>
</div>

<script>
// ── 세션 ────────────────────────────────────────────────────
let userId   = localStorage.getItem('fza_uid');
let userName = localStorage.getItem('fza_name');
let convId   = null;  // 현재 열린 대화 ID

if (userId && userName) showApp();

function startSession() {
  const name = document.getElementById('modal-name').value.trim();
  if (!name) return;
  if (!userId) userId = 'u' + Date.now().toString(36);
  userName = name;
  localStorage.setItem('fza_uid', userId);
  localStorage.setItem('fza_name', name);
  showApp();
  newChat();
}

function showApp() {
  document.getElementById('modal').style.display = 'none';
  document.getElementById('app').style.display = 'flex';
  document.getElementById('username-display').textContent = userName;
  document.getElementById('avatar').textContent = (userName||'?')[0].toUpperCase();
  loadConvList();
  fetchMemStatus();
}

function changeUser() {
  if (!confirm('계정을 변경하면 현재 세션이 초기화됩니다. 계속할까요?')) return;
  localStorage.clear();
  location.reload();
}

// ── 대화 목록 ────────────────────────────────────────────────
async function loadConvList() {
  try {
    const res = await fetch('/conversations?user_id=' + enc(userId));
    const list = await res.json();
    renderConvList(list);
    if (list.length === 0) newChat();
  } catch(e) { newChat(); }
}

function renderConvList(list) {
  const el = document.getElementById('conv-list');
  if (list.length === 0) {
    el.innerHTML = '<div style="font-size:11px;color:var(--text3);padding:12px 14px;font-style:italic">대화 기록이 없습니다</div>';
    return;
  }
  // 날짜 그룹 분리
  const today = new Date().toDateString();
  const yest  = new Date(Date.now()-86400000).toDateString();
  const groups = {today:[], yesterday:[], older:[]};
  for (const c of list) {
    const d = new Date(c.updated_at).toDateString();
    if (d === today) groups.today.push(c);
    else if (d === yest) groups.yesterday.push(c);
    else groups.older.push(c);
  }
  let html = '';
  if (groups.today.length)     html += renderGroup('오늘', groups.today);
  if (groups.yesterday.length) html += renderGroup('어제', groups.yesterday);
  if (groups.older.length)     html += renderGroup('이전', groups.older);
  el.innerHTML = html;
}

function renderGroup(label, items) {
  return `<div class="conv-group-title">${label}</div>` +
    items.map(c =>
      `<div class="conv-item${c.id===convId?' active':''}" onclick="openConv('${c.id}','${escAttr(c.title)}')" id="ci-${c.id}">
        <span class="conv-title">${escHtml(c.title)}</span>
        <div style="display:flex;align-items:center;gap:3px;flex-shrink:0">
          <span class="conv-turns">${c.turn_count}턴</span>
          <button class="conv-del-btn" onclick="event.stopPropagation();deleteConv('${c.id}')" title="삭제">✕</button>
        </div>
      </div>`
    ).join('');
}

async function newChat() {
  convId = null;
  document.getElementById('messages').innerHTML = '';
  document.getElementById('chat-title').textContent = '새 대화';
  // 기존 active 해제
  document.querySelectorAll('.conv-item.active').forEach(e => e.classList.remove('active'));
  addMsg('sys', `안녕하세요${userName ? ', ' + userName + '님' : ''}! 무엇이든 편하게 이야기해 주세요. 대화하면서 중요한 정보는 자동으로 기억합니다.`);
  document.getElementById('msg').focus();
}

async function openConv(id, title) {
  try {
    const res = await fetch('/conversation/' + id + '?user_id=' + enc(userId));
    const data = await res.json();
    convId = id;
    document.getElementById('chat-title').textContent = title;
    // 목록 active 업데이트
    document.querySelectorAll('.conv-item').forEach(e => {
      e.classList.toggle('active', e.id === 'ci-' + id);
    });
    // 메시지 복원
    const $msgs = document.getElementById('messages');
    $msgs.innerHTML = '';
    for (const m of data.history || []) {
      addMsg(m.role === 'user' ? 'user' : 'bot', m.content);
    }
    $msgs.scrollTop = $msgs.scrollHeight;
  } catch(e) { console.error(e); }
}

// ── 메시지 UI ────────────────────────────────────────────────
const $msgs = document.getElementById('messages');
function ts() { return new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'}); }

function renderMath(el) {
  if (!window._katexReady || typeof renderMathInElement === 'undefined') return;
  renderMathInElement(el, {
    delimiters: [
      {left: '$$',  right: '$$',  display: true},
      {left: '\\[', right: '\\]', display: true},
      {left: '$',   right: '$',   display: false},
      {left: '\\(', right: '\\)', display: false},
    ],
    throwOnError: false,
    output: 'html',
  });
}

function formatForBubble(rawText) {
  // LaTeX 구간은 KaTeX가 처리할 수 있도록 그대로 보존,
  // 나머지 텍스트는 HTML 이스케이프 후 줄바꿈만 처리.
  const mathPattern = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)/g;
  const isMath = /^(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)$/;
  return String(rawText).split(mathPattern).map(chunk => {
    if (!chunk) return '';
    if (isMath.test(chunk)) return escHtml(chunk);   // LaTeX: 이스케이프만, KaTeX가 렌더
    return escHtml(chunk).replace(/\n/g, '<br>');    // 일반 텍스트: 줄바꿈만
  }).join('');
}

function addMsg(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bub = document.createElement('div');
  bub.className = 'bubble';

  if (role === 'bot' || role === 'sys') {
    // 봇/시스템: 안전 변환 + 가독성 포맷 후 수식 렌더링
    bub.innerHTML = formatForBubble(text);
    // KaTeX가 이미 로드됐으면 즉시, 아니면 로드 후 렌더
    if (window._katexReady) {
      renderMath(bub);
    } else {
      document.addEventListener('katex-ready', () => renderMath(bub), {once: true});
      // 폴백: 500ms 후 재시도
      setTimeout(() => renderMath(bub), 500);
    }
  } else {
    // 유저: 그냥 텍스트
    bub.innerHTML = formatForBubble(text);
  }

  wrap.appendChild(bub);
  if (role !== 'sys') {
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = ts();
    wrap.appendChild(meta);
  }
  $msgs.appendChild(wrap);
  $msgs.scrollTop = $msgs.scrollHeight;
  return wrap;
}

function showMemToast(mems, replaced, wrap) {
  const hasNew  = mems && mems.length > 0;
  const hasRepl = replaced && replaced.length > 0;
  if (!hasNew && !hasRepl) return;

  const parts = [];
  const newOnly = mems ? mems.filter(m => !replaced || !replaced.some(r => r.new === m)) : [];
  if (newOnly.length)   parts.push(`✨ ${newOnly.length}개 기억 저장`);
  if (hasRepl)          parts.push(`🔄 ${replaced.length}개 기억 수정됨`);

  const t = document.createElement('div');
  t.className = 'mem-toast';
  t.textContent = parts.join('  ·  ');
  wrap.appendChild(t);

  if (hasNew) prependMemories(mems);
}

function showTyping() {
  const w = document.createElement('div');
  w.className = 'msg bot'; w.id = 'typing';
  w.innerHTML = '<div class="bubble typing"><span></span><span></span><span></span></div>';
  $msgs.appendChild(w); $msgs.scrollTop = $msgs.scrollHeight;
}
function removeTyping() { document.getElementById('typing')?.remove(); }

// ── 기억 패널 ────────────────────────────────────────────────
function toggleMem() {
  const body = document.getElementById('mem-body');
  const tog  = document.getElementById('mem-toggle');
  const open = body.classList.toggle('open');
  tog.classList.toggle('open', open);
}

let allMems = [];
function prependMemories(newOnes) {
  for (const m of newOnes) { if (!allMems.includes(m)) allMems.unshift(m); }
  renderMemList(newOnes.length);
}
function renderMemList(newCount = 0) {
  const el = document.getElementById('mem-list');
  if (!allMems.length) {
    el.innerHTML = '<div class="empty-hint">대화를 나누면 자동으로 기억됩니다</div>';
    return;
  }
  el.innerHTML = allMems.slice(0, 20).map((m, i) =>
    `<div class="mem-item${i < newCount ? ' new' : ''}">${escHtml(m)}</div>`
  ).join('');
}

async function fetchMemStatus() {
  if (!userId) return;
  try {
    const res = await fetch('/status?user_id=' + enc(userId));
    const d = await res.json();
    document.getElementById('s-rag').textContent = d.rag_count || 0;
    document.getElementById('s-mem').textContent = d.memories || 0;
    document.getElementById('pill-rag').className = 'pill' + (d.rag_active ? ' on' : '');
    if (d.memory_list && allMems.length === 0) {
      allMems = [...d.memory_list];
      renderMemList();
    }
  } catch(e) {}
}

// ── 전송 ────────────────────────────────────────────────────
const $msg = document.getElementById('msg');
const $send = document.getElementById('send');

$msg.addEventListener('input', () => {
  $msg.style.height = 'auto';
  $msg.style.height = Math.min($msg.scrollHeight, 120) + 'px';
});

function hint(prefix) { $msg.value = prefix; $msg.focus(); }

async function sendMessage() {
  const text = $msg.value.trim();
  if (!text || !userId) return;
  addMsg('user', text);
  $msg.value = ''; $msg.style.height = 'auto';
  $send.disabled = true;
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: text, user_id: userId, user_name: userName, conv_id: convId})
    });
    removeTyping();
    const data = await res.json();

    if (data.reply) {
      const wrap = addMsg('bot', data.reply);
      showMemToast(data.new_memories || [], data.replaced || [], wrap);
    }
    for (const line of data.messages || []) addMsg('bot', line);
    if (data.error) addMsg('bot', '❌ ' + data.error);

    // 새 대화 ID 확정 + 목록 갱신
    if (data.conv_id && data.conv_id !== convId) {
      convId = data.conv_id;
      loadConvList();
    } else if (data.conv_id) {
      loadConvList();
    }
    // 상단 제목 갱신
    if (data.title) document.getElementById('chat-title').textContent = data.title;
    fetchMemStatus();
  } catch(e) {
    removeTyping();
    addMsg('bot', '❌ 연결 오류: ' + e.message);
  }
  $send.disabled = false;
  $msg.focus();
}

$send.addEventListener('click', sendMessage);
$msg.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
document.getElementById('modal-name')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') startSession();
});

function enc(s) { return encodeURIComponent(s||''); }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escAttr(s) { return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }
function escId(s)   { return String(s).replace(/[^a-zA-Z0-9_-]/g,'_'); }

setInterval(fetchMemStatus, 8000);

// ── Memory Explorer ───────────────────────────────────────────
let meTab = 'mems';
let meData = { memories: [], profile: {} };

function openMemExplorer() {
  document.getElementById('me-modal').classList.add('open');
  loadMemExplorerData();
}
function closeMemExplorer() {
  document.getElementById('me-modal').classList.remove('open');
}

async function loadMemExplorerData() {
  if (!userId) return;
  try {
    const res = await fetch('/status?user_id=' + enc(userId));
    const d = await res.json();
    meData.memories = d.memory_list_all || [];
    meData.profile  = d.profile || {};
    renderMemExplorer();
  } catch(e) {}
}

function switchMemTab(tab) {
  meTab = tab;
  document.getElementById('tab-mems').classList.toggle('active', tab === 'mems');
  document.getElementById('tab-profile').classList.toggle('active', tab === 'profile');
  document.getElementById('me-add-section').style.display = tab === 'mems' ? '' : 'none';
  renderMemExplorer();
}

const CAT_COLORS = {
  '신상':'#22d3ee','직업':'#60a5fa','관계':'#f472b6',
  '목표':'#f59e0b','건강':'#10b981','취미':'#a78bfa','장소':'#fb923c','기타':'#64748b'
};
function catBadge(cat) {
  if (!cat) return '';
  const c = CAT_COLORS[cat] || '#64748b';
  return `<span class="cat-badge" style="background:${c}22;color:${c}">${cat}</span>`;
}
function fmtDate(ts) {
  if (!ts) return '';
  const diff = Math.floor((Date.now() - new Date(ts)) / 86400000);
  if (diff === 0) return '오늘';
  if (diff === 1) return '어제';
  if (diff < 7)  return diff + '일 전';
  if (diff < 30) return Math.floor(diff/7) + '주 전';
  return new Date(ts).toLocaleDateString('ko-KR');
}

function renderMemExplorer() {
  const q    = (document.getElementById('me-search-input')?.value || '').toLowerCase();
  const body = document.getElementById('me-body');
  document.getElementById('tab-mems-count').textContent    = '(' + meData.memories.length + ')';
  document.getElementById('tab-profile-count').textContent = '(' + Object.keys(meData.profile).length + ')';

  if (meTab === 'mems') {
    const pairs = [...meData.memories.entries()].reverse(); // [[realIdx, obj], ...]
    const filtered = q ? pairs.filter(([,m]) => (m.t||m).toLowerCase().includes(q)) : pairs;
    if (!filtered.length) {
      body.innerHTML = '<div class="me-empty">저장된 기억이 없습니다</div>';
      return;
    }
    body.innerHTML = '<div class="me-count">' + filtered.length + '개의 기억</div>' +
      filtered.map(([ri, m]) => {
        const txt = m.t || m;   // backward compat: string or object
        const ts  = m.ts  || null;
        const cat = m.cat || null;
        return `
        <div class="me-item" id="memi-${ri}">
          <div style="flex:1;min-width:0">
            <div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap">
              ${catBadge(cat)}
              <div class="me-item-text">${escHtml(txt)}</div>
            </div>
            ${ts ? `<div class="mem-ts">${fmtDate(ts)}</div>` : ''}
          </div>
          <div class="me-item-btns">
            <button class="me-btn" onclick="startEditMem(${ri})" title="수정">✏️</button>
            <button class="me-btn del" onclick="deleteMem(${ri})" title="삭제">🗑</button>
          </div>
        </div>
        <div class="me-edit-row" id="memi-edit-${ri}">
          <input class="me-edit-input" id="memi-inp-${ri}" value="${escAttr(txt)}"/>
          <button class="me-edit-save" onclick="saveMem(${ri})">저장</button>
          <button class="me-edit-cancel" onclick="cancelEditMem(${ri})">취소</button>
        </div>`;
      }).join('');

  } else {
    const entries = Object.entries(meData.profile);
    const filtered = q
      ? entries.filter(([k,v]) => k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q))
      : entries;
    if (!filtered.length) {
      body.innerHTML = '<div class="me-empty">저장된 프로필 정보가 없습니다</div>';
      return;
    }
    body.innerHTML = '<div class="me-count">' + filtered.length + '개의 항목</div>' +
      filtered.map(([k, v]) => {
        const id = escId(k);
        return `
          <div class="me-item" id="profi-${id}">
            <div class="me-item-key">${escHtml(k)}</div>
            <div class="me-item-text">${escHtml(String(v))}</div>
            <div class="me-item-btns">
              <button class="me-btn" onclick="startEditProf('${escAttr(k)}')" title="수정">✏️</button>
              <button class="me-btn del" onclick="deleteProf('${escAttr(k)}')" title="삭제">🗑</button>
            </div>
          </div>
          <div class="me-edit-row" id="profi-edit-${id}">
            <input class="me-edit-input" id="profi-inp-${id}" value="${escAttr(String(v))}"/>
            <button class="me-edit-save" onclick="saveProf('${escAttr(k)}')">저장</button>
            <button class="me-edit-cancel" onclick="cancelEditProf('${escAttr(id)}')">취소</button>
          </div>`;
      }).join('');
  }
}

// ── 기억 CRUD ────────────────────────────────────────────────
function startEditMem(ri) {
  document.getElementById('memi-edit-'+ri).classList.add('open');
  document.getElementById('memi-inp-'+ri).focus();
}
function cancelEditMem(ri) { document.getElementById('memi-edit-'+ri).classList.remove('open'); }

async function saveMem(ri) {
  const newText = document.getElementById('memi-inp-'+ri).value.trim();
  if (!newText) return;
  await fetch('/memory/edit', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_id: userId, index: ri, text: newText})});
  // update in-place (keep ts, update t)
  const old = meData.memories[ri];
  meData.memories[ri] = typeof old === 'object' ? {...old, t: newText} : {t: newText, ts: null, cat: null};
  renderMemExplorer();
  fetchMemStatus();
}

async function deleteMem(ri) {
  if (!confirm('이 기억을 삭제할까요?')) return;
  await fetch('/memory/delete', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_id: userId, index: ri})});
  meData.memories.splice(ri, 1);
  renderMemExplorer();
  fetchMemStatus();
}

async function submitAddMemory() {
  const text = document.getElementById('me-add-input').value.trim();
  if (!text) return;
  await fetch('/memory/add', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_id: userId, text: text})});
  document.getElementById('me-add-input').value = '';
  meData.memories.push({t: text, ts: new Date().toISOString().slice(0,19), cat: null});
  renderMemExplorer();
  fetchMemStatus();
}

// ── 프로필 CRUD ──────────────────────────────────────────────
function startEditProf(key) {
  const id = escId(key);
  document.getElementById('profi-edit-'+id).classList.add('open');
  document.getElementById('profi-inp-'+id).focus();
}
function cancelEditProf(id) { document.getElementById('profi-edit-'+id).classList.remove('open'); }

async function saveProf(key) {
  const id  = escId(key);
  const val = document.getElementById('profi-inp-'+id).value.trim();
  await fetch('/profile/edit', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_id: userId, key: key, value: val})});
  meData.profile[key] = val;
  renderMemExplorer();
}

async function deleteProf(key) {
  if (!confirm('\'' + key + '\' 항목을 삭제할까요?')) return;
  await fetch('/profile/delete', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_id: userId, key: key})});
  delete meData.profile[key];
  renderMemExplorer();
  fetchMemStatus();
}

// ── 모달 외부 클릭으로 닫기 ──────────────────────────────────
document.getElementById('me-modal').addEventListener('click', e => {
  if (e.target.id === 'me-modal') closeMemExplorer();
});
document.getElementById('me-add-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') submitAddMemory();
});

// ── 대화 삭제 ────────────────────────────────────────────────
async function deleteConv(id) {
  if (!confirm('이 대화를 삭제할까요?')) return;
  await fetch('/conversation/delete', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({user_id: userId, conv_id: id})});
  if (convId === id) { convId = null; newChat(); }
  loadConvList();
}

// ── 음성 입력 (Web Speech API) ───────────────────────────────
let recognition = null;
let isListening = false;

function toggleVoice() {
  if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
    addMsg('sys', '이 브라우저는 음성 입력을 지원하지 않습니다. Chrome을 사용해주세요.');
    return;
  }
  isListening ? stopVoice() : startVoice();
}

function startVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'ko-KR';
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.onresult = e => {
    const transcript = e.results[0][0].transcript;
    $msg.value = ($msg.value + ' ' + transcript).trim();
    $msg.dispatchEvent(new Event('input'));
  };
  recognition.onend = () => stopVoice();
  recognition.onerror = () => stopVoice();
  recognition.start();
  isListening = true;
  document.getElementById('voice-btn').classList.add('listening');
  document.getElementById('voice-btn').title = '음성 인식 중... (클릭하여 중지)';
}

function stopVoice() {
  if (recognition) { try { recognition.stop(); } catch(e){} recognition = null; }
  isListening = false;
  document.getElementById('voice-btn').classList.remove('listening');
  document.getElementById('voice-btn').title = '음성 입력';
}

// ── 파일 업로드 ──────────────────────────────────────────────
function triggerFileUpload() {
  document.getElementById('file-input').click();
}

async function handleFileUpload(input) {
  const file = input.files[0];
  if (!file) return;
  input.value = '';   // 재선택 가능하도록 초기화

  const hint = document.getElementById('upload-hint');
  hint.textContent = `📎 ${file.name} 처리 중...`;
  hint.classList.add('visible');

  const reader = new FileReader();
  reader.onload = async () => {
    const dataUrl = reader.result;
    const comma   = dataUrl.indexOf(',');
    const b64     = comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
    const mime    = file.type || 'text/plain';

    try {
      showTyping();
      const res = await fetch('/upload', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_id: userId, filename: file.name, content: b64, mime: mime, conv_id: convId})
      });
      removeTyping();
      const data = await res.json();
      hint.classList.remove('visible');

      if (data.summary) {
        addMsg('sys', `📎 ${file.name} 요약 완료`);
        const wrap = addMsg('bot', data.summary);
        showMemToast(data.new_memories || [], data.replaced || [], wrap);
        if (data.new_memories?.length || data.replaced?.length) fetchMemStatus();
      }
      if (data.error) addMsg('bot', '❌ ' + data.error);
    } catch(e) {
      removeTyping();
      hint.classList.remove('visible');
      addMsg('bot', '❌ 업로드 오류: ' + e.message);
    }
  };

  // 이미지는 DataURL로, 나머지(텍스트/PDF)도 DataURL로 읽기
  reader.readAsDataURL(file);
}
</script>
</body>
</html>"""


# ── 슬래시 명령 ──────────────────────────────────────────────
def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return [l for l in buf.getvalue().splitlines() if l.strip()]


def handle_slash(text: str, manager: FZAManager) -> dict | None:
    lower = text.lower()

    if lower.startswith("/기억") or lower == "/memory":
        mems = manager.bridge.user_profile.get('_memories', [])
        profile = {k: v for k, v in manager.bridge.user_profile.items() if k != '_memories'}
        lines = []
        if profile:
            lines.append("👤 프로필:")
            for k, v in profile.items():
                lines.append(f"  · {k}: {v}")
        if mems:
            lines.append(f"💡 일반 기억 (총 {len(mems)}개, 최근 20개):")
            for m in mems[-20:]:
                lines.append(f"  · {m}")
        return {"messages": lines or ["아직 저장된 기억이 없습니다."]}

    if lower.startswith("/수식 추가 "):
        rest = text[7:].strip().split(None, 1)
        if len(rest) < 2:
            return {"messages": ["형식: /수식 추가 [이름] [수식]"]}
        lines = _capture(manager.add_formula, rest[0], rest[1])
        return {"messages": lines or [f"✅ '{rest[0]}' 등록됨."]}

    if lower in ("/수식 목록", "/formula"):
        lines = _capture(manager.list_formulas)
        return {"messages": lines or ["등록된 수식이 없습니다."]}

    if lower in ("/초기화", "/clear"):
        lines = _capture(manager.flush_memory)
        return {"messages": lines or ["✅ 대화 기록을 초기화했습니다."]}

    if lower in ("/상태", "/status"):
        mems = manager.bridge.user_profile.get('_memories', [])
        return {"messages": [
            f"🧠 RAG: {'활성 (' + str(len(manager.rag)) + '개)' if manager.rag else '비활성'}",
            f"💡 일반 기억: {len(mems)}개",
            f"📐 수식: {len(manager.math_engine.math_vault)}개",
            f"💬 대화 턴: {len(manager.bridge.conversation_history) // 2}회",
        ]}

    if lower in ("/잠금", "/lock"):
        return {"messages": _capture(manager.lock_and_export) or ["✅ 잠금 완료."]}

    return None


def get_status(user_id: str) -> dict:
    m = get_manager(user_id)
    mems  = m.bridge.user_profile.get('_memories', [])
    dates = m.bridge.user_profile.get('_memory_dates', [])
    cats  = m.bridge.user_profile.get('_memory_cats', [])
    profile = {k: v for k, v in m.bridge.user_profile.items() if not k.startswith('_')}
    memory_list_all = [
        {"t": t, "ts": dates[i] if i < len(dates) else None,
         "cat": cats[i] if i < len(cats) else None}
        for i, t in enumerate(mems)
    ]
    return {
        "rag_count":       len(m.rag) if m.rag else 0,
        "rag_active":      m.rag is not None,
        "memories":        len(mems),
        "memory_list":     list(reversed(mems[-20:])),   # sidebar preview (strings)
        "memory_list_all": memory_list_all,              # explorer: full objects
        "profile":         profile,
        "formulas":        list(m.math_engine.math_vault.keys()),
    }


# ── HTTP 핸들러 ───────────────────────────────────────────────
class FZAWebHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query)
        uid = qs.get("user_id", ["default"])[0]

        if p.path == "/":
            self._html(WEB_PAGE.encode())
        elif p.path == "/status":
            self._json(get_status(uid))
        elif p.path == "/conversations":
            convs = get_manager(uid).list_conversations()
            self._json(convs)
        elif p.path.startswith("/conversation/"):
            conv_id = p.path.split("/")[-1]
            manager = get_manager(uid)
            data = manager.load_conversation(conv_id)
            self._json(data)
        elif p.path == "/manifest.json":
            manifest = {
                "name": "FZA — 망각 없는 AI",
                "short_name": "FZA",
                "description": "당신의 모든 것을 기억하는 AI",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0a0a12",
                "theme_color": "#0a0a12",
                "icons": [
                    {"src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌿</text></svg>",
                     "sizes": "any", "type": "image/svg+xml"}
                ]
            }
            body = json.dumps(manifest, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/manifest+json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception as exc:
            self._json({"error": str(exc)})
            return

        # ── /chat ────────────────────────────────────────────────
        if p.path == "/chat":
            try:
                uid       = payload.get("user_id", "default")
                user_name = payload.get("user_name", "")
                message   = payload.get("message", "").strip()
                conv_id   = payload.get("conv_id") or None

                if not message:
                    self._json({"messages": ["빈 메시지입니다."]})
                    return

                manager = get_manager(uid)

                # 첫 방문: 이름 저장
                if user_name and "이름" not in manager.bridge.user_profile:
                    manager.bridge.set_user_fact("이름", user_name)
                    manager.bridge.save_profile()

                # 슬래시 명령
                if message.startswith("/"):
                    result = handle_slash(message, manager)
                    if result:
                        self._json(result)
                        return

                # 일반 대화
                result = manager.chat_and_remember(message, conv_id=conv_id)
                title = None
                if result["conv_id"]:
                    conv_path = os.path.join(
                        manager.conversations_path,
                        result["conv_id"] + ".json"
                    )
                    if os.path.exists(conv_path):
                        with open(conv_path) as f:
                            title = json.load(f).get("title")
                self._json({
                    "reply":        result["reply"],
                    "new_memories": result["new_memories"],
                    "replaced":     result.get("replaced", []),
                    "conv_id":      result["conv_id"],
                    "title":        title,
                })
            except Exception as exc:
                self._json({"error": str(exc)})

        # ── /memory/delete ────────────────────────────────────────
        elif p.path == "/memory/delete":
            uid   = payload.get("user_id", "default")
            index = payload.get("index", -1)
            m = get_manager(uid)
            ok = m.bridge.delete_memory(int(index))
            if ok:
                m.bridge.save_profile()
            self._json({"ok": ok})

        # ── /memory/edit ──────────────────────────────────────────
        elif p.path == "/memory/edit":
            uid   = payload.get("user_id", "default")
            index = payload.get("index", -1)
            text  = payload.get("text", "").strip()
            m = get_manager(uid)
            ok = m.bridge.edit_memory(int(index), text)
            if ok:
                m.bridge.save_profile()
            self._json({"ok": ok})

        # ── /memory/add ───────────────────────────────────────────
        elif p.path == "/memory/add":
            uid  = payload.get("user_id", "default")
            text = payload.get("text", "").strip()
            m = get_manager(uid)
            if text:
                m.bridge.add_memory(text)
                m.bridge.save_profile()
                if m.rag:
                    m.rag.add(text)
                    m.rag.save(path=os.path.join(m.vault_path, "rag_memory"))
                self._json({"ok": True})
            else:
                self._json({"ok": False})

        # ── /profile/delete ───────────────────────────────────────
        elif p.path == "/profile/delete":
            uid = payload.get("user_id", "default")
            key = payload.get("key", "")
            m = get_manager(uid)
            ok = m.bridge.delete_profile_key(key)
            if ok:
                m.bridge.save_profile()
            self._json({"ok": ok})

        # ── /profile/edit ─────────────────────────────────────────
        elif p.path == "/profile/edit":
            uid   = payload.get("user_id", "default")
            key   = payload.get("key", "")
            value = payload.get("value", "")
            m = get_manager(uid)
            ok = m.bridge.edit_profile_key(key, value)
            if ok:
                m.bridge.save_profile()
            self._json({"ok": ok})

        # ── /conversation/delete ──────────────────────────────────
        elif p.path == "/conversation/delete":
            uid     = payload.get("user_id", "default")
            conv_id = payload.get("conv_id", "")
            m = get_manager(uid)
            ok = m.delete_conversation(conv_id)
            self._json({"ok": ok})

        # ── /upload ───────────────────────────────────────────────
        elif p.path == "/upload":
            try:
                uid      = payload.get("user_id", "default")
                filename = payload.get("filename", "file")
                content  = payload.get("content", "")
                mime     = payload.get("mime", "text/plain")
                conv_id  = payload.get("conv_id") or None

                m = get_manager(uid)
                summary = m.bridge.process_file(filename, content, mime)

                # 파일 내용을 대화에 포함시켜 기억 추출
                new_facts = m.bridge.auto_extract_memory(
                    f"[파일 업로드: {filename}]", summary)
                added, replaced = [], []
                if new_facts:
                    added, replaced = m.bridge.smart_merge_memories(new_facts)
                    for fact in added:
                        if m.rag: m.rag.add(fact)
                    for pair in replaced:
                        if m.rag: m.rag.add(pair["new"])
                    if added or replaced:
                        m.bridge.save_profile()
                        if m.rag:
                            m.rag.save(path=os.path.join(m.vault_path, "rag_memory"))

                self._json({
                    "summary":      summary,
                    "new_memories": added + [p2["new"] for p2 in replaced],
                    "replaced":     replaced,
                })
            except Exception as exc:
                self._json({"error": str(exc)})

        else:
            self.send_error(404)

    def _html(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


# ── 서버 시작 ─────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FZAWebHandler)
    print(f"\n🌿 FZA System — 망각 없는 AI")
    print(f"   웹 UI: http://127.0.0.1:{args.port}")
    print(f"   종료: Ctrl+C\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
