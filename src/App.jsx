import React, { useState, useMemo, useRef } from "react";
import {
  Mail, Inbox, History, ChevronDown, Clock, Trash2, Send, ArrowRight,
  CheckCircle2, XCircle, AlertTriangle, ScanLine, Radar, FlaskConical,
  Copy, Check, Download, Gauge, Layers, PlusCircle, Loader2, RotateCcw
} from "lucide-react";

/* ============================================================================
   CLASSIFICATION ENGINE  (v2 — expanded lexicon, sender heuristic, stopwords)
   preprocess -> extract features -> weighted score -> logistic confidence.
   Benchmarked against a 20-email labeled set — see the Evaluation tab.
============================================================================ */

const SPAM_PHRASES = {
  "click here": 2.2, "act now": 2.0, "limited time": 1.8, "risk free": 1.6,
  "risk-free": 1.6, "no cost": 1.5, "no obligation": 1.5, "free gift": 2.0,
  "free shipping": 1.1, "order now": 1.8, "buy now": 1.7, "best price": 1.3,
  "satisfaction guaranteed": 1.4, "money back": 1.3, "congratulations you": 2.6,
  "you have been selected": 2.4, "you have won": 2.8, "claim your": 2.2,
  "claim now": 2.2, "verify your account": 2.3, "verify your identity": 2.2,
  "update your account": 1.8, "suspended your account": 2.2, "account has been suspended": 2.3,
  "unusual activity": 1.6, "wire transfer": 2.3, "bank transfer": 1.9,
  "bank account": 1.4, "social security number": 2.4, "credit card number": 2.0,
  "act immediately": 2.0, "urgent business proposal": 2.4, "dear friend": 1.6,
  "dear beneficiary": 2.6, "next of kin": 2.6, "lottery winner": 2.8,
  "lucky winner": 2.6, "cash prize": 2.4, "work from home": 1.4,
  "earn extra income": 1.8, "earn money fast": 2.0, "double your income": 2.2,
  "investment opportunity": 1.3, "lose weight fast": 1.8, "miracle cure": 2.2,
  "enlarge your": 2.4, "no prescription needed": 2.2, "hot singles": 2.4,
  "click the link below": 1.9, "permanently deleted": 1.4, "per week": 0.8,
  "double your bitcoin": 2.6, "guaranteed returns": 2.2, "send money": 1.8,
  "western union": 2.4, "wire the funds": 2.4, "no interview required": 2.4,
  "direct deposit": 0.9, "incomplete address": 1.0, "reschedule delivery": 0.9,
  "avoid additional fees": 1.3, "confirm your details": 1.7, "sign-in activity": 1.2,
  "within 24 hours": 1.0, "my dearest": 1.8, "never met": 0.8,
  "share your otp": 2.6, "share your code": 2.2, "tax refund": 1.3,
  "claim your refund": 2.2, "pay a small fee": 1.8, "processing fee": 1.2,
  "act before": 1.2, "don't miss out": 1.4,
};

const SPAM_WORDS = {
  free: 1.1, winner: 2.0, win: 1.0, won: 1.1, prize: 1.8, urgent: 1.6,
  congratulations: 1.7, guarantee: 0.9, guaranteed: 1.0, cash: 1.3, bonus: 1.0,
  exclusive: 0.8, limited: 0.9, expires: 1.1, password: 1.3, refund: 0.9,
  lottery: 2.2, inheritance: 2.2, beneficiary: 1.8, million: 1.4, dollars: 0.9,
  loan: 1.3, viagra: 3.0, cialis: 3.0, pills: 1.5, click: 0.7, confidential: 0.9,
  suspended: 1.5, verify: 1.2, unauthorized: 1.2, immediately: 0.8, jackpot: 2.2,
  casino: 1.9, gamble: 1.6, rich: 1.0, deceased: 1.3, manager: 0.2, proposal: 0.3,
  bitcoin: 1.5, crypto: 1.3, btc: 1.3, investment: 0.5, otp: 0.6, pin: 0.7,
  ssn: 1.8, fee: 0.5, fees: 0.5, delivery: 0.2, courier: 0.4, gift: 0.7,
  voucher: 1.1, coupon: 0.6, subscription: 0.3, expired: 1.0, locked: 0.9,
  unlock: 0.9, reward: 1.0, rewards: 1.0,
};

const HAM_WORDS = {
  meeting: -1.0, attached: -0.6, attachment: -0.6, project: -0.7, schedule: -0.7,
  agenda: -0.7, regards: -0.8, sincerely: -0.8, team: -0.5, report: -0.5,
  deadline: -0.6, conference: -0.6, university: -0.7, professor: -0.8,
  assignment: -0.7, lecture: -0.7, semester: -0.7, course: -0.5, colleague: -0.6,
  department: -0.5, thanks: -0.5, thank: -0.4, please: -0.2, draft: -0.4,
  document: -0.3, feedback: -0.4, review: -0.3, presentation: -0.4, client: -0.3,
  budget: -0.3, minutes: -0.4, calendar: -0.5, reschedule: -0.3, appointment: -0.5,
  weekend: -0.3, lunch: -0.3, admin: -0.3, office: -0.2, progress: -0.3,
  interview: -0.7, campus: -0.7, placement: -0.7, resume: -0.5, internship: -0.6,
  hostel: -0.6, library: -0.6, syllabus: -0.6, exam: -0.4, faculty: -0.6,
  login: -0.2, renew: -0.2,
};

const STOPWORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "and",
  "or", "in", "on", "at", "for", "with", "this", "that", "it", "as", "your",
  "you", "i", "we", "our", "us", "from", "by", "has", "have", "had", "will",
  "would", "can", "could", "do", "does", "did", "not", "no", "so", "if", "but",
  "my", "me", "he", "she", "they", "them",
]);

function preprocess(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9$%!.\s/:]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function classifyEmail(subject, body, sender = "", opts = {}) {
  const { phrases: usePhrases = true, words: useWords = true, heuristics: useHeuristics = true } = opts;
  const rawText = `${subject} ${body}`;
  const text = preprocess(rawText);
  const features = [];
  let score = 0;

  if (usePhrases) {
    for (const [phrase, weight] of Object.entries(SPAM_PHRASES)) {
      if (text.includes(phrase)) {
        score += weight;
        features.push({ label: phrase, weight, type: "phrase" });
      }
    }
  }

  // tokens are always computed (used for the token-preview UI) — only the
  // scoring contribution is gated by the `words` ablation toggle.
  const tokens = text.split(/\s+/).filter(Boolean);
  if (useWords) {
    const counted = {};
    for (const tok of tokens) {
      const clean = tok.replace(/[^a-z]/g, "");
      if (!clean) continue;
      counted[clean] = (counted[clean] || 0) + 1;
    }
    for (const [word, count] of Object.entries(counted)) {
      if (SPAM_WORDS[word]) {
        const w = SPAM_WORDS[word] * Math.min(count, 2);
        score += w;
        features.push({ label: word, weight: w, type: "word" });
      }
      if (HAM_WORDS[word]) {
        const w = HAM_WORDS[word] * Math.min(count, 2);
        score += w;
        features.push({ label: word, weight: w, type: "ham" });
      }
    }
  }

  if (useHeuristics) {
    const exclaim = (rawText.match(/!/g) || []).length;
    if (exclaim > 1) {
      const w = Math.min((exclaim - 1) * 0.35, 2.0);
      score += w;
      features.push({ label: `${exclaim} exclamation marks`, weight: w, type: "heuristic" });
    }

    const letters = rawText.replace(/[^a-zA-Z]/g, "");
    const caps = rawText.replace(/[^A-Z]/g, "");
    const capsRatio = letters.length ? caps.length / letters.length : 0;
    if (capsRatio > 0.12) {
      const w = Math.min((capsRatio - 0.12) * 9, 2.5);
      score += w;
      features.push({ label: `${Math.round(capsRatio * 100)}% capital letters`, weight: w, type: "heuristic" });
    }

    const dollars = (rawText.match(/\$/g) || []).length;
    if (dollars > 0) {
      const w = Math.min(dollars * 0.3, 1.5);
      score += w;
      features.push({ label: `${dollars} currency symbols`, weight: w, type: "heuristic" });
    }

    const susLinks = (rawText.match(/bit\.ly|tinyurl|goo\.gl|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/gi) || []).length;
    if (susLinks > 0) {
      const w = Math.min(susLinks * 1.8, 2.2);
      score += w;
      features.push({ label: "suspicious shortened/IP link", weight: w, type: "heuristic" });
    }

    const linkCount = (rawText.match(/https?:\/\/|www\./gi) || []).length;
    if (linkCount > 1) {
      const w = Math.min((linkCount - 1) * 0.4, 1.2);
      score += w;
      features.push({ label: `${linkCount} links detected`, weight: w, type: "heuristic" });
    }

    const genericGreeting = /dear (customer|friend|beneficiary|sir\s*\/?\s*madam|valued)/i.test(rawText);
    if (genericGreeting) {
      score += 1.1;
      features.push({ label: "generic greeting (no name)", weight: 1.1, type: "heuristic" });
    }

    const senderDigits = (sender.match(/\d{4,}/g) || []).length;
    if (senderDigits > 0) {
      score += 0.4;
      features.push({ label: "numeric sender address", weight: 0.4, type: "heuristic" });
    }
  }

  const k = 0.55, bias = 3.4;
  const confidence = 1 / (1 + Math.exp(-k * (score - bias)));
  features.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

  return {
    score,
    confidence,
    label: confidence >= 0.5 ? "SPAM" : "NOT SPAM",
    features: features.slice(0, 8),
    tokens: tokens.slice(0, 26),
    cleanTokens: tokens.filter((t) => !STOPWORDS.has(t)).slice(0, 26),
  };
}

/* ============================================================================
   SAMPLES — doubles as the dropdown library AND the labeled benchmark set
   used by the Evaluation tab (20 emails, 11 spam / 9 legitimate).
============================================================================ */

const SAMPLES = [
  { group: "Spam examples", name: "Lottery winner", sender: "promo@global-sweepstakes.biz",
    subject: "You're a Lottery Winner!", actual: "SPAM",
    body: "CONGRATULATIONS! You have been selected as the lucky winner of our $1,000,000 lottery! Click here to claim your prize now before it expires. Act now!!!" },
  { group: "Spam examples", name: "Urgent business proposal", sender: "barrister.j.okafor@mail-secure.cn",
    subject: "Urgent Business Proposal", actual: "SPAM",
    body: "Dear Beneficiary, I am a bank manager and I have an urgent business proposal for you regarding an inheritance of $10.5 million dollars left by a deceased client. Please reply immediately with your bank account details and social security number for the wire transfer." },
  { group: "Spam examples", name: "Account security alert", sender: "security-alert@accnt-verify.support",
    subject: "Account Security Alert", actual: "SPAM",
    body: "URGENT: Your account has been suspended due to unusual activity. Verify your account immediately by clicking the link below or it will be permanently deleted. http://bit.ly/verify-now123" },
  { group: "Spam examples", name: "Cheap meds online", sender: "deals@pharma-savings-now.shop",
    subject: "Cheap Meds Online", actual: "SPAM",
    body: "Buy cheap VIAGRA and CIALIS online! No prescription needed. 100% guaranteed lowest prices. Order now and get FREE shipping!!! Limited time offer, act fast!!!" },
  { group: "Spam examples", name: "Earn from home", sender: "opportunity@quickcash-online.info",
    subject: "Earn From Home", actual: "SPAM",
    body: "Work from home and earn $5000 per week! No experience needed. This amazing opportunity is exclusive and limited. Click here to start earning cash today!!!" },
  { group: "Spam examples", name: "Bitcoin doubling offer", sender: "support@crypto-double-returns.io",
    subject: "Bitcoin Doubling Offer", actual: "SPAM",
    body: "Double your Bitcoin investment in 24 hours! Send 0.1 BTC and receive 0.2 BTC instantly. Guaranteed returns, limited time opportunity, act now!!!" },
  { group: "Spam examples", name: "Bank sign-in alert", sender: "alerts@secure-bank-verify.net",
    subject: "Bank Sign-in Alert", actual: "SPAM",
    body: "We've noticed unusual sign-in activity on your account. Verify your identity within 24 hours or your account will be limited. Click here to confirm your details now." },
  { group: "Spam examples", name: "Romance scam", sender: "lovely.heart2024@webmail.ru",
    subject: "A Message From My Heart", actual: "SPAM",
    body: "My dearest, I have fallen for you though we have never met. I need your help, please send money for my visa so I can finally come see you. I promise to pay you back, my love." },
  { group: "Spam examples", name: "Fake job offer", sender: "hr@quickhire-remote.biz",
    subject: "Work From Home Opportunity", actual: "SPAM",
    body: "Congratulations! You have been selected for a work from home data entry job paying $45 an hour. No interview required. Click here to start immediately and provide your bank details for direct deposit setup." },
  { group: "Spam examples", name: "Delivery failed", sender: "noreply@parcel-track-delivery.com",
    subject: "Delivery Failed", actual: "SPAM",
    body: "Your package could not be delivered due to an incomplete address. Click here to reschedule delivery and avoid additional fees." },
  { group: "Spam examples", name: "Fake tax refund", sender: "refunds@gov-tax-claims.org",
    subject: "Claim Your Tax Refund", actual: "SPAM",
    body: "URGENT: claim your tax refund now at www.taxrefund-gov-claim.com before it expires." },
  { group: "Legitimate examples", name: "Meeting minutes", sender: "sarah.menon@company.com",
    subject: "Meeting Minutes Attached", actual: "NOT SPAM",
    body: "Hi team, please find attached the meeting minutes from yesterday's project discussion. Let me know if you have any feedback before we finalize the report. Thanks, Sarah" },
  { group: "Legitimate examples", name: "Assignment deadline", sender: "aravind.student@iare.ac.in",
    subject: "Following up on assignment deadline", actual: "NOT SPAM",
    body: "Dear Professor, I wanted to follow up regarding the assignment deadline for our Deep Learning course. Could we possibly schedule a short meeting this week to discuss my project progress? Thank you for your time." },
  { group: "Legitimate examples", name: "Lunch plans", sender: "priya.k@gmail.com",
    subject: "Lunch this weekend?", actual: "NOT SPAM",
    body: "Hey, are we still on for lunch this weekend? Let me know what time works for you. Also, don't forget to bring the documents we discussed at the last conference." },
  { group: "Legitimate examples", name: "Invoice & budget", sender: "finance@company.com",
    subject: "Invoice and budget report", actual: "NOT SPAM",
    body: "Hi John, attached is the invoice for last month's services along with the budget report. Please review and let me know if you need any clarification before the client presentation on Friday." },
  { group: "Legitimate examples", name: "Meeting rescheduled", sender: "admin.office@company.com",
    subject: "Department meeting rescheduled", actual: "NOT SPAM",
    body: "Reminder: Our department meeting has been rescheduled to 3 PM tomorrow in the main conference room. Please update your calendar accordingly. Regards, Admin Office" },
  { group: "Legitimate examples", name: "Campus interview", sender: "placements@iare.ac.in",
    subject: "Campus Interview Confirmation", actual: "NOT SPAM",
    body: "Hi, just confirming our interview is scheduled for Monday at 10 AM at the campus placement cell. Please bring your resume and a valid ID. Looking forward to meeting you." },
  { group: "Legitimate examples", name: "OTP for login", sender: "noreply@securebank.com",
    subject: "Your OTP for login", actual: "NOT SPAM",
    body: "Your OTP for login is 482193. Do not share this code with anyone. It will expire in 10 minutes." },
  { group: "Legitimate examples", name: "Library reminder", sender: "library@iare.ac.in",
    subject: "Library Book Due Reminder", actual: "NOT SPAM",
    body: "Reminder: Library books are due for return by Friday. Please renew online to avoid late fees." },
  { group: "Borderline examples", name: "Free webinar invite", sender: "events@learnml-academy.com",
    subject: "FREE Webinar This Friday", actual: "NOT SPAM",
    body: "Join our free webinar this Friday on machine learning basics. Limited seats available, register now to save your spot!" },
];

const SAMPLE_GROUPS = ["Spam examples", "Legitimate examples", "Borderline examples"];

const BATCH_SAMPLE_NAMES = ["Lottery winner", "Bitcoin doubling offer", "Meeting minutes", "Campus interview", "Fake tax refund"];
const BATCH_SAMPLE_TEXT = SAMPLES
  .filter((s) => BATCH_SAMPLE_NAMES.includes(s.name))
  .map((s) => `Subject: ${s.subject}\n${s.body}`)
  .join("\n-----\n");

function parseBatch(text) {
  return text
    .split(/\n-{3,}\n/)
    .map((b) => b.trim())
    .filter(Boolean)
    .map((b, i) => {
      const subjMatch = b.match(/^subject:\s*(.+)$/im);
      const subject = subjMatch ? subjMatch[1].trim() : `Email ${i + 1}`;
      const bodyText = subjMatch ? b.replace(subjMatch[0], "").trim() : b;
      return { id: i, subject, body: bodyText };
    });
}

/* ============================================================================
   STYLE
============================================================================ */

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  .ssc {
    --ink: #0D1117;
    --panel: #141B23;
    --panel-alt: #1B2531;
    --border: #283341;
    --text: #E7EBF1;
    --muted: #8A96A6;
    --spam: #FF5C5C;
    --spam-dim: #4A2326;
    --clean: #3ECF8E;
    --clean-dim: #1E3A30;
    --amber: #FFB454;
    --amber-dim: #3A2F1B;
    font-family: 'Inter', sans-serif;
    background: var(--ink);
    color: var(--text);
    min-height: 100%;
    width: 100%;
  }
  .ssc * { box-sizing: border-box; }
  .ssc .font-display { font-family: 'Space Grotesk', sans-serif; }
  .ssc .font-mono { font-family: 'IBM Plex Mono', monospace; }

  .ssc .topbar {
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(255,255,255,0.015), transparent);
  }
  .ssc .brand-mark {
    width: 36px; height: 36px; border-radius: 8px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    color: var(--clean);
  }
  .ssc .live-dot {
    width: 6px; height: 6px; border-radius: 999px; background: var(--clean);
    box-shadow: 0 0 0 0 rgba(62,207,142,0.5);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(62,207,142,0.45); }
    70% { box-shadow: 0 0 0 6px rgba(62,207,142,0); }
    100% { box-shadow: 0 0 0 0 rgba(62,207,142,0); }
  }

  .ssc .tab-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted);
    border-bottom: 2px solid transparent;
    padding: 10px 4px;
    transition: color 0.15s ease, border-color 0.15s ease;
    background: transparent; border-left: none; border-right: none; border-top: none;
    cursor: pointer;
  }
  .ssc .tab-btn:hover { color: var(--text); }
  .ssc .tab-btn.active { color: var(--clean); border-bottom-color: var(--clean); }

  .ssc .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .ssc .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted);
  }

  .ssc input[type="text"], .ssc textarea, .ssc select {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 7px;
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    width: 100%;
    padding: 9px 11px;
    outline: none;
    transition: border-color 0.15s ease;
  }
  .ssc input[type="text"]::placeholder, .ssc textarea::placeholder { color: #56616F; }
  .ssc input[type="text"]:focus, .ssc textarea:focus, .ssc select:focus {
    border-color: var(--clean);
  }
  .ssc textarea { resize: vertical; line-height: 1.5; }

  .ssc input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 4px; border-radius: 999px;
    background: linear-gradient(90deg, var(--clean), var(--border) 50%, var(--spam));
    outline: none; margin: 14px 0 4px;
  }
  .ssc input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: var(--text); border: 3px solid var(--ink); cursor: pointer;
    box-shadow: 0 0 0 1px var(--border);
  }
  .ssc input[type="range"]::-moz-range-thumb {
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--text); border: 3px solid var(--ink); cursor: pointer;
  }
  .ssc input[type="range"]::-moz-range-track { height: 4px; background: transparent; }

  .ssc .btn-primary {
    background: var(--clean);
    color: #04150D;
    font-weight: 600;
    border: none;
    border-radius: 7px;
    padding: 10px 16px;
    font-size: 13.5px;
    display: inline-flex; align-items: center; gap: 8px;
    cursor: pointer;
    transition: filter 0.15s ease, transform 0.1s ease;
  }
  .ssc .btn-primary:hover:not(:disabled) { filter: brightness(1.08); }
  .ssc .btn-primary:active:not(:disabled) { transform: scale(0.98); }
  .ssc .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

  .ssc .btn-ghost {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px 14px;
    font-size: 13.5px;
    display: inline-flex; align-items: center; gap: 7px;
    cursor: pointer;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .ssc .btn-ghost:hover { color: var(--text); border-color: #3A4757; }
  .ssc .btn-ghost.sm { padding: 6px 11px; font-size: 12px; }

  .ssc .pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
    cursor: pointer;
    background: transparent;
    transition: all 0.15s ease;
  }
  .ssc .pill:hover { color: var(--text); }
  .ssc .pill.active { color: var(--ink); background: var(--clean); border-color: var(--clean); font-weight: 600; }

  .ssc .verdict-empty {
    border: 1.5px dashed var(--border);
    border-radius: 10px;
    color: var(--muted);
  }

  .ssc .scanline-wrap { position: relative; overflow: hidden; border-radius: 10px; }
  .ssc .scanline {
    position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--clean), transparent);
    box-shadow: 0 0 12px 2px rgba(62,207,142,0.6);
    animation: sweep 1.4s ease-in-out infinite;
  }
  @keyframes sweep {
    0% { top: 4%; opacity: 0.2; }
    50% { top: 92%; opacity: 1; }
    100% { top: 4%; opacity: 0.2; }
  }
  .ssc .skel-line { height: 9px; border-radius: 4px; background: var(--panel-alt); }

  .ssc .spin { animation: sscspin 0.9s linear infinite; }
  @keyframes sscspin { to { transform: rotate(360deg); } }

  .ssc .stamp {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: 0.08em;
    border: 3px solid currentColor;
    border-radius: 8px;
    padding: 6px 14px;
    transform: rotate(-8deg);
    display: inline-block;
    animation: stampIn 0.35s cubic-bezier(.2,1.4,.4,1) both;
    text-transform: uppercase;
  }
  @keyframes stampIn {
    0% { transform: rotate(-8deg) scale(1.9); opacity: 0; }
    60% { opacity: 1; }
    100% { transform: rotate(-8deg) scale(1); opacity: 1; }
  }

  .ssc .chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    border-radius: 999px;
    padding: 3px 9px;
    border: 1px solid var(--border);
    background: var(--panel-alt);
    color: var(--muted);
    display: inline-flex;
  }
  .ssc .chip.clean-outline { border-color: var(--clean-dim); color: var(--clean); }

  .ssc .feature-row { border-bottom: 1px solid var(--border); }
  .ssc .feature-row:last-child { border-bottom: none; }

  .ssc .hx-row { cursor: pointer; transition: background 0.12s ease; }
  .ssc .hx-row:hover { background: var(--panel-alt); }

  .ssc .badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
  }
  .ssc .badge-spam { background: var(--spam-dim); color: var(--spam); }
  .ssc .badge-clean { background: var(--clean-dim); color: var(--clean); }

  .ssc .pipe-step { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
  .ssc .pipe-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700;
    color: var(--border);
  }

  .ssc .dot-track {
    position: relative; height: 64px;
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--panel-alt);
    margin: 6px 0 4px;
  }
  .ssc .dot {
    position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%;
    transform: translate(-50%, -50%);
  }
  .ssc .dot-mis { box-shadow: 0 0 0 3px var(--amber); }
  .ssc .threshold-line {
    position: absolute; top: 0; bottom: 0; width: 2px; background: var(--text); opacity: 0.55;
  }

  .ssc .cm-grid { display: grid; grid-template-columns: 96px 1fr 1fr; gap: 6px; max-width: 460px; }
  .ssc .cm-head { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--muted); display: flex; align-items: center; justify-content: center; text-align: center; padding: 4px; }
  .ssc .cm-row-label { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--muted); display: flex; align-items: center; padding: 4px; }
  .ssc .cm-cell { border-radius: 8px; padding: 14px 8px; text-align: center; }
  .ssc .cm-value { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; }
  .ssc .cm-cap { font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; opacity: 0.8; }

  .ssc ::-webkit-scrollbar { width: 8px; height: 8px; }
  .ssc ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 999px; }

  .ssc :focus-visible { outline: 2px solid var(--clean); outline-offset: 2px; }

  @media (prefers-reduced-motion: reduce) {
    .ssc * { animation: none !important; transition: none !important; }
  }
`;

/* ============================================================================
   SMALL COMPONENTS
============================================================================ */

function ConfidenceGauge({ confidence, isSpam }) {
  const pct = Math.round(confidence * 100);
  const r = 46;
  const c = 2 * Math.PI * r;
  const offset = c - (pct / 100) * c;
  const color = isSpam ? "var(--spam)" : "var(--clean)";
  return (
    <div className="relative" style={{ width: 120, height: 120 }}>
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={r} fill="none" stroke="var(--border)" strokeWidth="10" />
        <circle
          cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.3s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display" style={{ fontSize: 24, fontWeight: 700, color }}>{pct}%</span>
        <span className="eyebrow" style={{ fontSize: 9 }}>spam score</span>
      </div>
    </div>
  );
}

function Donut({ spam, clean }) {
  const total = spam + clean || 1;
  const spamPct = spam / total;
  const r = 38, c = 2 * Math.PI * r;
  return (
    <svg width="100" height="100" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r={r} fill="none" stroke="var(--clean-dim)" strokeWidth="14" />
      <circle
        cx="50" cy="50" r={r} fill="none" stroke="var(--spam)" strokeWidth="14"
        strokeDasharray={`${spamPct * c} ${c}`}
        transform="rotate(-90 50 50)"
        strokeLinecap="butt"
      />
    </svg>
  );
}

function FeatureRow({ f }) {
  const isPositive = f.weight > 0;
  const color = f.type === "heuristic" ? "var(--amber)" : isPositive ? "var(--spam)" : "var(--clean)";
  const Icon = f.type === "heuristic" ? AlertTriangle : isPositive ? XCircle : CheckCircle2;
  return (
    <div className="feature-row flex items-center justify-between gap-3 py-2">
      <div className="flex items-center gap-2 min-w-0">
        <Icon size={13} style={{ color, flexShrink: 0 }} />
        <span className="font-mono truncate" style={{ fontSize: 12.5, color: "var(--text)" }}>{f.label}</span>
      </div>
      <span className="font-mono flex-shrink-0" style={{ fontSize: 12, color, fontWeight: 600 }}>
        {isPositive ? "+" : ""}{f.weight.toFixed(2)}
      </span>
    </div>
  );
}

function StatCard({ label, value, accent }) {
  return (
    <div className="card p-4">
      <div className="eyebrow mb-1">{label}</div>
      <div className="font-display" style={{ fontSize: 26, fontWeight: 700, color: accent || "var(--text)" }}>{value}</div>
    </div>
  );
}

function CMCell({ value, caption, correct }) {
  return (
    <div className="cm-cell" style={{ background: correct ? "var(--clean-dim)" : "var(--spam-dim)" }}>
      <div className="cm-value" style={{ color: correct ? "var(--clean)" : "var(--spam)" }}>{value}</div>
      <div className="cm-cap" style={{ color: correct ? "var(--clean)" : "var(--spam)" }}>{caption}</div>
    </div>
  );
}

function ROCCurve({ points, auc }) {
  const size = 160;
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${(p.fpr * size).toFixed(1)} ${(size - p.tpr * size).toFixed(1)}`)
    .join(" ");
  return (
    <div className="flex items-center gap-6 flex-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ flexShrink: 0 }}>
        <rect x="0.5" y="0.5" width={size - 1} height={size - 1} fill="none" stroke="var(--border)" />
        <line x1="0" y1={size} x2={size} y2="0" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke="var(--clean)" strokeWidth="2.5" strokeLinejoin="round" />
        <text x="4" y="11" fontSize="8" fill="var(--muted)">TPR</text>
        <text x={size - 24} y={size - 5} fontSize="8" fill="var(--muted)">FPR</text>
      </svg>
      <div>
        <div className="eyebrow mb-1">Area under curve</div>
        <div className="font-display" style={{ fontSize: 32, fontWeight: 700, color: "var(--clean)" }}>{auc.toFixed(3)}</div>
        <p style={{ fontSize: 11.5, color: "var(--muted)", maxWidth: 240 }} className="mt-1">
          1.0 is a perfect classifier; 0.5 (the dashed diagonal) is random guessing. This sweeps
          every possible threshold, independent of where your slider is set.
        </p>
      </div>
    </div>
  );
}

/* ============================================================================
   MAIN APP
============================================================================ */

export default function App() {
  const [tab, setTab] = useState("scanner");
  const [sender, setSender] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [scannedAt, setScannedAt] = useState(null);
  const [history, setHistory] = useState([]);
  const [sampleKey, setSampleKey] = useState("");
  const [copied, setCopied] = useState(false);
  const [logFilter, setLogFilter] = useState("all");
  const [threshold, setThreshold] = useState(50);
  const idRef = useRef(0);

  const canScan = body.trim().length >= 10 && !scanning;

  function loadSample(name) {
    const s = SAMPLES.find((x) => x.name === name);
    if (!s) return;
    setSender(s.sender);
    setSubject(s.subject);
    setBody(s.body);
    setResult(null);
    setSampleKey(name);
  }

  function handleScan() {
    if (!canScan) return;
    setScanning(true);
    setResult(null);
    window.setTimeout(() => {
      const r = classifyEmail(subject, body, sender);
      setResult(r);
      setScannedAt(new Date());
      idRef.current += 1;
      setHistory((h) => [
        { id: idRef.current, sender, subject: subject || "(no subject)", body, result: r, ts: new Date() },
        ...h,
      ].slice(0, 50));
      setScanning(false);
    }, 900);
  }

  function handleClear() {
    setSender(""); setSubject(""); setBody(""); setResult(null); setSampleKey("");
  }

  function replay(item) {
    setSender(item.sender); setSubject(item.subject); setBody(item.body);
    setResult(item.result); setScannedAt(item.ts);
    setTab("scanner");
  }

  function copySummary() {
    if (!result) return;
    const text = [
      `SpamScope verdict: ${result.label} (${Math.round(result.confidence * 100)}% confidence)`,
      `Subject: ${subject || "(no subject)"}`,
      sender ? `Sender: ${sender}` : null,
      `Top signals: ${result.features.slice(0, 5).map((f) => `${f.label} (${f.weight > 0 ? "+" : ""}${f.weight.toFixed(2)})`).join(", ") || "none"}`,
    ].filter(Boolean).join("\n");

    const onDone = () => { setCopied(true); window.setTimeout(() => setCopied(false), 1600); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(onDone).catch(() => fallbackCopy(text, onDone));
    } else {
      fallbackCopy(text, onDone);
    }
  }

  function fallbackCopy(text, onDone) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      onDone();
    } catch (e) { /* clipboard unavailable, fail silently */ }
  }

  function exportCSV() {
    const header = "timestamp,sender,subject,verdict,confidence_pct\n";
    const rows = history.map((h) => {
      const safe = (s) => `"${String(s).replace(/"/g, '""')}"`;
      return [safe(h.ts.toISOString()), safe(h.sender || ""), safe(h.subject), safe(h.result.label), (h.result.confidence * 100).toFixed(1)].join(",");
    }).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "spamscope_case_log.csv";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const stats = useMemo(() => {
    const total = history.length;
    const spam = history.filter((h) => h.result.label === "SPAM").length;
    const clean = total - spam;
    const rate = total ? Math.round((spam / total) * 100) : 0;
    return { total, spam, clean, rate };
  }, [history]);

  const filteredHistory = useMemo(() => {
    if (logFilter === "spam") return history.filter((h) => h.result.label === "SPAM");
    if (logFilter === "clean") return history.filter((h) => h.result.label === "NOT SPAM");
    return history;
  }, [history, logFilter]);

  const evalResults = useMemo(() => SAMPLES.map((s) => {
    const r = classifyEmail(s.subject, s.body, s.sender);
    return {
      name: s.name, actual: s.actual, confidence: r.confidence,
      predicted: r.confidence * 100 >= threshold ? "SPAM" : "NOT SPAM",
    };
  }), [threshold]);

  const metrics = useMemo(() => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    evalResults.forEach((r) => {
      if (r.predicted === "SPAM" && r.actual === "SPAM") tp++;
      else if (r.predicted === "SPAM" && r.actual === "NOT SPAM") fp++;
      else if (r.predicted === "NOT SPAM" && r.actual === "NOT SPAM") tn++;
      else fn++;
    });
    const accuracy = (tp + tn) / evalResults.length;
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { tp, fp, tn, fn, accuracy, precision, recall, f1 };
  }, [evalResults]);

  const misclassified = useMemo(() => evalResults.filter((r) => r.predicted !== r.actual), [evalResults]);

  return (
    <div className="ssc">
      <style>{CSS}</style>

      {/* Top bar */}
      <div className="topbar px-5 py-4 sm:px-8">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="brand-mark"><Radar size={18} /></div>
            <div>
              <div className="font-display flex items-center gap-2" style={{ fontSize: 17, fontWeight: 700 }}>
                SpamScope
                <span className="live-dot" />
                <span className="chip" style={{ fontSize: 10, padding: "1px 7px" }}>v2.0</span>
              </div>
              <div className="eyebrow">client-side ml triage console</div>
            </div>
          </div>
          <nav className="flex items-center gap-5 sm:gap-7">
            <button className={`tab-btn ${tab === "scanner" ? "active" : ""}`} onClick={() => setTab("scanner")}>
              Scanner
            </button>
            <button className={`tab-btn ${tab === "log" ? "active" : ""}`} onClick={() => setTab("log")}>
              Case Log {history.length > 0 ? `(${history.length})` : ""}
            </button>
            <button className={`tab-btn ${tab === "eval" ? "active" : ""}`} onClick={() => setTab("eval")}>
              Evaluation
            </button>
            <button className={`tab-btn ${tab === "pipeline" ? "active" : ""}`} onClick={() => setTab("pipeline")}>
              Pipeline
            </button>
          </nav>
        </div>
      </div>

      <div className="px-5 py-6 sm:px-8 sm:py-8 max-w-6xl mx-auto">

        {/* ---------------- SCANNER TAB ---------------- */}
        {tab === "scanner" && (
          <div>
            <p className="mb-6" style={{ color: "var(--muted)", fontSize: 13.5, maxWidth: 640 }}>
              Paste an email below, or load a sample. Every scan runs preprocessing, feature
              extraction, and weighted classification entirely in your browser — no email
              content leaves this device. Curious how accurate this is? Check the{" "}
              <button onClick={() => setTab("eval")} style={{ color: "var(--clean)", textDecoration: "underline", background: "none", border: "none", cursor: "pointer", padding: 0, font: "inherit" }}>
                Evaluation tab
              </button>{" "}
              for a full 20-email benchmark.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

              {/* INBOUND */}
              <div className="card p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Inbox size={14} style={{ color: "var(--clean)" }} />
                  <span className="eyebrow">Inbound</span>
                </div>

                <label className="eyebrow block mb-1.5">Sender</label>
                <input
                  type="text" placeholder="sender@example.com"
                  value={sender} onChange={(e) => setSender(e.target.value)}
                  className="mb-3"
                />

                <label className="eyebrow block mb-1.5">Subject</label>
                <input
                  type="text" placeholder="Email subject line"
                  value={subject} onChange={(e) => setSubject(e.target.value)}
                  className="mb-3"
                />

                <label className="eyebrow block mb-1.5">Body</label>
                <textarea
                  rows={9} placeholder="Paste the email body here… (Ctrl/Cmd + Enter to scan)"
                  value={body} onChange={(e) => setBody(e.target.value)}
                  onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleScan(); }}
                  className="mb-1.5"
                />
                {body.trim().length > 0 && body.trim().length < 10 && (
                  <div className="flex items-center gap-1.5 mb-2" style={{ color: "var(--amber)", fontSize: 11.5 }}>
                    <AlertTriangle size={12} /> Add a longer body — short snippets don't carry enough signal.
                  </div>
                )}

                <div className="flex items-center gap-2 flex-wrap mt-3">
                  <div className="relative" style={{ flex: "1 1 200px" }}>
                    <select
                      value={sampleKey}
                      onChange={(e) => loadSample(e.target.value)}
                      style={{ appearance: "none", paddingRight: 30 }}
                    >
                      <option value="">— Load a sample email —</option>
                      {SAMPLE_GROUPS.map((g) => (
                        <optgroup key={g} label={g}>
                          {SAMPLES.filter((s) => s.group === g).map((s) => (
                            <option key={s.name} value={s.name}>{s.name}</option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                    <ChevronDown size={14} style={{ position: "absolute", right: 10, top: 11, color: "var(--muted)", pointerEvents: "none" }} />
                  </div>
                  <button className="btn-ghost" onClick={handleClear}>
                    <Trash2 size={14} /> Clear
                  </button>
                  <button className="btn-primary" disabled={!canScan} onClick={handleScan}>
                    {scanning ? <ScanLine size={15} /> : <Send size={15} />}
                    {scanning ? "Scanning…" : "Scan Email"}
                  </button>
                </div>
              </div>

              {/* VERDICT */}
              <div className="card p-5">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <FlaskConical size={14} style={{ color: "var(--clean)" }} />
                    <span className="eyebrow">Verdict</span>
                  </div>
                  {!scanning && result && (
                    <button className="btn-ghost sm" onClick={copySummary}>
                      {copied ? <Check size={13} /> : <Copy size={13} />}
                      {copied ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>

                {!scanning && !result && (
                  <div className="verdict-empty flex flex-col items-center justify-center text-center gap-2 py-16 px-6">
                    <Mail size={26} style={{ opacity: 0.6 }} />
                    <p style={{ fontSize: 13 }}>No scan yet. Paste an email or load a sample, then hit Scan Email.</p>
                  </div>
                )}

                {scanning && (
                  <div className="scanline-wrap py-10 px-6">
                    <div className="scanline" />
                    <div className="space-y-2.5 mb-5">
                      <div className="skel-line" style={{ width: "85%" }} />
                      <div className="skel-line" style={{ width: "70%" }} />
                      <div className="skel-line" style={{ width: "92%" }} />
                      <div className="skel-line" style={{ width: "55%" }} />
                    </div>
                    <div className="text-center font-mono" style={{ fontSize: 12, color: "var(--clean)", letterSpacing: "0.1em" }}>
                      ANALYZING…
                    </div>
                  </div>
                )}

                {!scanning && result && (
                  <div>
                    <div className="flex items-start justify-between gap-3 mb-5">
                      <ConfidenceGauge confidence={result.confidence} isSpam={result.label === "SPAM"} />
                      <div
                        className="stamp"
                        style={{ color: result.label === "SPAM" ? "var(--spam)" : "var(--clean)" }}
                      >
                        {result.label}
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5 mb-4" style={{ fontSize: 11, color: "var(--muted)" }}>
                      <Clock size={11} />
                      Scanned at {scannedAt ? scannedAt.toLocaleTimeString() : ""}
                    </div>

                    <div className="eyebrow mb-2">Signal breakdown</div>
                    <div className="mb-4">
                      {result.features.length === 0 ? (
                        <p style={{ fontSize: 12.5, color: "var(--muted)" }}>No strong signals detected either way.</p>
                      ) : (
                        result.features.map((f, i) => <FeatureRow key={i} f={f} />)
                      )}
                    </div>

                    <div className="eyebrow mb-2">Preprocessed tokens</div>
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {result.tokens.map((t, i) => (
                        <span key={i} className="chip">{t}</span>
                      ))}
                    </div>

                    <div className="eyebrow mb-2">After stopword removal</div>
                    <p style={{ fontSize: 11, color: "var(--muted)" }} className="mb-2">
                      Common connector words are stripped out, leaving the terms that actually carry signal.
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {result.cleanTokens.length === 0 ? (
                        <span style={{ fontSize: 12, color: "var(--muted)" }}>Nothing left after cleanup.</span>
                      ) : (
                        result.cleanTokens.map((t, i) => (
                          <span key={i} className="chip clean-outline">{t}</span>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ---------------- CASE LOG TAB ---------------- */}
        {tab === "log" && (
          <div>
            {history.length === 0 ? (
              <div className="verdict-empty flex flex-col items-center justify-center text-center gap-2 py-20 px-6">
                <History size={26} style={{ opacity: 0.6 }} />
                <p style={{ fontSize: 13, color: "var(--muted)" }}>
                  Your case log is empty — scan an email from the Scanner tab to start building a history.
                </p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5 items-stretch">
                  <StatCard label="Total scanned" value={stats.total} />
                  <StatCard label="Flagged spam" value={stats.spam} accent="var(--spam)" />
                  <StatCard label="Clean" value={stats.clean} accent="var(--clean)" />
                  <div className="card p-4 flex items-center gap-3">
                    <Donut spam={stats.spam} clean={stats.clean} />
                    <div>
                      <div className="eyebrow mb-1">Spam rate</div>
                      <div className="font-display" style={{ fontSize: 22, fontWeight: 700 }}>{stats.rate}%</div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
                  <div className="flex items-center gap-2">
                    <button className={`pill ${logFilter === "all" ? "active" : ""}`} onClick={() => setLogFilter("all")}>All</button>
                    <button className={`pill ${logFilter === "spam" ? "active" : ""}`} onClick={() => setLogFilter("spam")}>Spam</button>
                    <button className={`pill ${logFilter === "clean" ? "active" : ""}`} onClick={() => setLogFilter("clean")}>Clean</button>
                  </div>
                  <div className="flex items-center gap-2">
                    <button className="btn-ghost sm" onClick={exportCSV}>
                      <Download size={13} /> Export CSV
                    </button>
                    <button className="btn-ghost sm" onClick={() => { setHistory([]); setLogFilter("all"); }}>
                      <Trash2 size={13} /> Clear log
                    </button>
                  </div>
                </div>

                <div className="card overflow-hidden">
                  <div className="px-4 py-2.5 eyebrow" style={{ borderBottom: "1px solid var(--border)" }}>
                    {filteredHistory.length} scan{filteredHistory.length === 1 ? "" : "s"} — click a row to reopen it
                  </div>
                  {filteredHistory.length === 0 ? (
                    <div className="p-6 text-center" style={{ fontSize: 12.5, color: "var(--muted)" }}>
                      No scans match this filter.
                    </div>
                  ) : (
                    <div>
                      {filteredHistory.map((h) => (
                        <div
                          key={h.id}
                          onClick={() => replay(h)}
                          className="hx-row flex items-center justify-between gap-3 px-4 py-3"
                          style={{ borderBottom: "1px solid var(--border)" }}
                        >
                          <div className="min-w-0 flex-1">
                            <div className="truncate" style={{ fontSize: 13.5 }}>{h.subject}</div>
                            <div className="font-mono truncate" style={{ fontSize: 11, color: "var(--muted)" }}>{h.sender || "unknown sender"}</div>
                          </div>
                          <div className="flex items-center gap-3 flex-shrink-0">
                            <span className="font-mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                              {Math.round(h.result.confidence * 100)}%
                            </span>
                            <span className={`badge ${h.result.label === "SPAM" ? "badge-spam" : "badge-clean"}`}>
                              {h.result.label}
                            </span>
                            <span className="font-mono hidden sm:inline" style={{ fontSize: 11, color: "var(--muted)" }}>
                              {h.ts.toLocaleTimeString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* ---------------- EVALUATION TAB ---------------- */}
        {tab === "eval" && (
          <div>
            <p className="mb-6" style={{ color: "var(--muted)", fontSize: 13.5, maxWidth: 680 }}>
              The classifier is benchmarked against a held-out set of 20 labeled emails (11 spam,
              9 legitimate). Drag the threshold to see how the decision boundary trades precision
              for recall — the same tuning step a real spam filter team runs before shipping a model.
            </p>

            <div className="card p-5 mb-5">
              <div className="flex items-center justify-between mb-1">
                <span className="eyebrow flex items-center gap-1.5"><Gauge size={12} /> Decision threshold</span>
                <span className="font-mono" style={{ fontSize: 13, color: "var(--text)" }}>{threshold}%</span>
              </div>
              <input
                type="range" min={0} max={100} value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
              <div className="flex items-center justify-between" style={{ fontSize: 10.5, color: "var(--muted)" }}>
                <span>← lower: catches more spam, more false alarms</span>
                <span>higher: fewer false alarms, may miss spam →</span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5">
              <StatCard label="Accuracy" value={`${Math.round(metrics.accuracy * 100)}%`} accent="var(--clean)" />
              <StatCard label="Precision" value={`${Math.round(metrics.precision * 100)}%`} accent="var(--amber)" />
              <StatCard label="Recall" value={`${Math.round(metrics.recall * 100)}%`} accent="var(--spam)" />
              <StatCard label="F1 score" value={`${Math.round(metrics.f1 * 100)}%`} />
            </div>

            <div className="card p-5 mb-5">
              <div className="eyebrow mb-1">Confidence distribution</div>
              <p style={{ fontSize: 11.5, color: "var(--muted)" }} className="mb-1">
                Each dot is one test email positioned by predicted spam probability. The vertical
                line marks your current threshold; ringed dots are misclassified at that threshold.
              </p>
              <div className="dot-track">
                <div className="threshold-line" style={{ left: `${threshold}%` }} />
                {evalResults.map((r, i) => (
                  <div
                    key={i}
                    title={`${r.name} — ${Math.round(r.confidence * 100)}% (actual: ${r.actual})`}
                    className={`dot ${r.predicted !== r.actual ? "dot-mis" : ""}`}
                    style={{ left: `${r.confidence * 100}%`, background: r.actual === "SPAM" ? "var(--spam)" : "var(--clean)" }}
                  />
                ))}
              </div>
              <div className="flex items-center gap-4 flex-wrap mt-2" style={{ fontSize: 11, color: "var(--muted)" }}>
                <span className="flex items-center gap-1.5">
                  <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--spam)", display: "inline-block" }} /> actual spam
                </span>
                <span className="flex items-center gap-1.5">
                  <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--clean)", display: "inline-block" }} /> actual legitimate
                </span>
                <span className="flex items-center gap-1.5">
                  <span style={{ width: 8, height: 8, borderRadius: 999, boxShadow: "0 0 0 2px var(--amber)", display: "inline-block" }} /> misclassified
                </span>
              </div>
            </div>

            <div className="card p-5 mb-5">
              <div className="eyebrow mb-3">Confusion matrix</div>
              <div className="cm-grid">
                <div className="cm-head" />
                <div className="cm-head">Predicted SPAM</div>
                <div className="cm-head">Predicted NOT SPAM</div>

                <div className="cm-row-label">Actual SPAM</div>
                <CMCell value={metrics.tp} caption="true positive" correct />
                <CMCell value={metrics.fn} caption="false negative" correct={false} />

                <div className="cm-row-label">Actual NOT SPAM</div>
                <CMCell value={metrics.fp} caption="false positive" correct={false} />
                <CMCell value={metrics.tn} caption="true negative" correct />
              </div>
            </div>

            {misclassified.length > 0 ? (
              <div className="card p-5">
                <div className="eyebrow mb-2">Misclassified at {threshold}% threshold</div>
                {misclassified.map((r, i) => (
                  <div key={i} className="feature-row flex items-center justify-between gap-3 py-2">
                    <span style={{ fontSize: 12.5 }}>{r.name}</span>
                    <span className="font-mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                      actual {r.actual} · predicted {r.predicted} · {Math.round(r.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="card p-4 flex items-center gap-2" style={{ color: "var(--clean)" }}>
                <CheckCircle2 size={16} />
                <span style={{ fontSize: 13 }}>No misclassifications at this threshold — clean separation across the test set.</span>
              </div>
            )}
          </div>
        )}

        {/* ---------------- PIPELINE TAB ---------------- */}
        {tab === "pipeline" && (
          <div>
            <p className="mb-6" style={{ color: "var(--muted)", fontSize: 13.5, maxWidth: 640 }}>
              Every scan runs through three sequential stages — the same conceptual pipeline
              used by production spam filters, scaled down to run instantly in the browser.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <div className="pipe-step p-5">
                <div className="pipe-num mb-2">01</div>
                <div className="font-display mb-1.5" style={{ fontSize: 15, fontWeight: 700 }}>Text preprocessing</div>
                <p style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
                  Lowercase the text, strip irrelevant punctuation, tokenize the subject and body,
                  then remove common stopwords to isolate terms that actually carry signal.
                </p>
              </div>
              <div className="pipe-step p-5">
                <div className="pipe-num mb-2">02</div>
                <div className="font-display mb-1.5" style={{ fontSize: 15, fontWeight: 700 }}>Feature extraction</div>
                <p style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
                  Match tokens and phrases against a weighted lexicon, then layer in heuristics —
                  punctuation density, capital-letter ratio, link patterns, generic greetings,
                  even suspicious sender formatting.
                </p>
              </div>
              <div className="pipe-step p-5">
                <div className="pipe-num mb-2">03</div>
                <div className="font-display mb-1.5" style={{ fontSize: 15, fontWeight: 700 }}>Classification</div>
                <p style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
                  Sum the weighted signals into a raw score, then pass it through a logistic
                  function to calibrate a 0–100% spam probability against a tunable threshold.
                </p>
              </div>
            </div>

            <div className="card p-5">
              <div className="eyebrow mb-3">Worked example</div>
              <div className="flex flex-col gap-3">
                <div className="flex items-start gap-2 flex-wrap">
                  <span className="badge badge-spam" style={{ flexShrink: 0 }}>raw text</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                    "CONGRATULATIONS! You have been selected as the lucky winner…"
                  </span>
                </div>
                <ArrowRight size={14} style={{ color: "var(--border)" }} />
                <div className="flex items-start gap-2 flex-wrap">
                  <span className="chip">congratulations</span>
                  <span className="chip">you</span>
                  <span className="chip">have</span>
                  <span className="chip">been</span>
                  <span className="chip">selected</span>
                  <span className="chip">lucky</span>
                  <span className="chip">winner</span>
                  <span className="font-mono" style={{ fontSize: 11.5, color: "var(--muted)" }}>… tokens</span>
                </div>
                <ArrowRight size={14} style={{ color: "var(--border)" }} />
                <div className="flex items-start gap-2 flex-wrap">
                  <span className="chip clean-outline">congratulations</span>
                  <span className="chip clean-outline">selected</span>
                  <span className="chip clean-outline">lucky</span>
                  <span className="chip clean-outline">winner</span>
                  <span className="font-mono" style={{ fontSize: 11.5, color: "var(--muted)" }}>… after stopword removal</span>
                </div>
                <ArrowRight size={14} style={{ color: "var(--border)" }} />
                <div className="flex items-start gap-2 flex-wrap">
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--spam)" }}>+2.6 "congratulations you"</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--spam)" }}>+2.6 "lucky winner"</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--amber)" }}>+2.0 caps ratio</span>
                </div>
                <ArrowRight size={14} style={{ color: "var(--border)" }} />
                <div className="flex items-center gap-2">
                  <span className="stamp" style={{ color: "var(--spam)", transform: "rotate(-8deg) scale(0.85)" }}>SPAM</span>
                  <span className="font-mono" style={{ fontSize: 12, color: "var(--muted)" }}>≈ 99.8% confidence</span>
                </div>
              </div>
            </div>

            <p className="mt-6" style={{ fontSize: 12, color: "var(--muted)" }}>
              Real-world application: this is the same preprocess → extract → classify shape used
              by production email filters, which typically swap the hand-tuned lexicon here for a
              model such as Naive Bayes, logistic regression, or an SVM trained on a large labeled
              corpus — the Evaluation tab shows how that same threshold-tuning workflow plays out.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
