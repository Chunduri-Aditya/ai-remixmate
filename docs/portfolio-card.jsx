/**
 * AI RemixMate — Portfolio Project Card (React Component)
 *
 * Drop into any React portfolio site:
 *   import RemixMateCard from './portfolio-card';
 *   <RemixMateCard />
 *
 * Uses Tailwind CSS classes. If your portfolio doesn't use Tailwind,
 * use the portfolio-card.html version instead.
 *
 * Props (all optional):
 *   githubUrl     — override GitHub repo URL
 *   accessUrl     — override access request URL
 *   showStatus    — show/hide "actively maintained" badge (default: true)
 */

import { useState } from "react";

const TAGS = [
  "Python", "FastAPI", "PyTorch", "Demucs", "librosa",
  "FAISS", "RAG", "DSP", "Streamlit"
];

const STATS = [
  { value: "14+", label: "Core Modules" },
  { value: "80+", label: "Tests" },
  { value: "20+", label: "API Endpoints" },
];

const HIGHLIGHTS = [
  "Beat-grid locked transitions with sample-level precision",
  "GPU-accelerated STFT, cosine similarity, time-stretching (MPS/CUDA)",
  "ITU-R BS.1770-4 loudness mastering with true-peak limiting",
  "RAG vector search for natural-language music discovery",
  "Overnight batch pipeline with sleep prevention (681+ song library)",
];

export default function RemixMateCard({
  githubUrl = "https://github.com/Chunduri-Aditya/ai-remixmate",
  accessUrl = "https://github.com/Chunduri-Aditya/ai-remixmate/issues/new?template=access-request.yml",
  showStatus = true,
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      className={`bg-[#12121c] border border-[#1e1e2e] rounded-2xl max-w-md w-full overflow-hidden transition-all duration-300 ${
        hovered ? "translate-y-[-4px] shadow-[0_12px_40px_rgba(192,132,252,0.15)]" : ""
      }`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Header */}
      <div className="bg-gradient-to-br from-[#1a1028] to-[#0f0f1a] p-6 relative overflow-hidden">
        <div className="absolute -top-1/2 -left-1/2 w-[200%] h-[200%] bg-radial-gradient opacity-60 animate-pulse" />
        <div className="text-3xl mb-3 relative">🎛️</div>
        <h3 className="text-xl font-bold relative bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent">
          AI RemixMate
        </h3>
        <p className="text-gray-500 text-sm mt-1 relative">
          AI-Powered DJ Engine & Audio Intelligence Platform
        </p>
      </div>

      {/* Body */}
      <div className="p-6">
        <p className="text-gray-400 text-sm leading-relaxed mb-5">
          Full-stack audio engineering system that analyzes tracks, plans DJ transitions at musical
          phrase boundaries, renders stem-aware crossfades with dynamic EQ, and powers a RAG-based
          music discovery engine. GPU-accelerated on Apple Silicon and NVIDIA.
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mb-5">
          {TAGS.map((tag) => (
            <span
              key={tag}
              className="bg-purple-500/10 border border-purple-500/20 text-purple-400 px-2.5 py-1 rounded-full text-xs font-medium"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-5">
          {STATS.map((stat) => (
            <div key={stat.label} className="text-center p-2.5 bg-white/[0.03] rounded-lg">
              <div className="text-lg font-bold text-purple-400">{stat.value}</div>
              <div className="text-[0.65rem] text-gray-500 uppercase tracking-wider mt-0.5">
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        {/* Highlights */}
        <ul className="mb-5 space-y-1.5">
          {HIGHLIGHTS.map((item, i) => (
            <li key={i} className="flex items-center gap-2 text-gray-400 text-[0.82rem]">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 flex-shrink-0" />
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Actions */}
      <div className="flex gap-3 px-6 pb-6">
        <a
          href={accessUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold text-center bg-gradient-to-br from-purple-400 to-indigo-400 text-black hover:brightness-110 transition-all"
        >
          Request Access
        </a>
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold text-center border border-[#1e1e2e] text-gray-200 hover:border-purple-400 hover:text-purple-400 transition-all"
        >
          View on GitHub
        </a>
      </div>

      {/* Footer */}
      {showStatus && (
        <div className="border-t border-[#1e1e2e] px-6 py-3 flex justify-between items-center">
          <div className="flex items-center gap-1.5 text-green-400 text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            Actively maintained
          </div>
          <div className="text-gray-500 text-[0.7rem] flex items-center gap-1">
            🔒 Private repo
          </div>
        </div>
      )}
    </div>
  );
}
