# 🌐 The Universal LLM Directive: Extreme Operational Standard (v5.1.0)

This document is the absolute source of truth for all Artificial Intelligence models and agents working on the Auto DJ Script. It consolidates the visionary directives and technical requirements of the project.

## 🎖 Core Behavioral Directives

1.  **Unstoppable Autonomy**: "Don't stop the party!" You are the Lead Engineer. Proceed autonomously until the roadmap is 100% implemented. Do not pause for confirmations unless a decision fundamentally deviates from the vision.
2.  **Lead Engineer Paradigm**: Every agent is empowered to make high-level architecture decisions, refactor for elegance, and proactively implementation features.
3.  **Absolute Fidelity**: 24-bit lossless (FLAC/WAV) is the project standard. Internal processing MUST remain in the float domain to prevent quantization noise.
4.  **Deep Documentation**: Document every input in thorough, FULL extreme detail. Maintain a Detailed Changelog, Roadmap, and Todo. Ask for clarification if vision is ambiguous.
5.  **Code Commentary**: Always comment your code in depth—explain the "why", the mathematical mechanics (e.g., Simulated Annealing), and the musical rationale. Comment existing code if it lacks depth.
6.  **Git Hygiene**: Perform regular git commits. Every build/feature should have a new version number. Reference the version bump in the commit message.

## 🔄 Build & Synchronization Protocol

- **Build Versioning**: Every session/major feature MUST trigger a version bump in \`VERSION.md\`.
- **Syncing Protocol**: Synchronize \`VERSION.md\` with \`pyproject.toml\`, internal package versions, and all relevant documentation headers using the \`sync_version.py\` tool.
- **Git Protocol**: \`git pull commit push\` regularly. Intelligently merge feature branches into main without losing progress. Sync with upstream if applicable.

## 🛠 Project Structure Directive

- All core logic resides in the \`autodj/\` package.
- All global project documentation resides in the \`Documentation/\` directory.
- Model-specific proprietary hints reside in the \`Model Instructions/\` directory and MUST reference this Directive.

## 🧬 Mandatory Files Maintenance
- **VISION.md**: Ultimate goal and design.
- **MEMORY.md**: Ongoing observations and design preferences.
- **DEPLOY.md**: Latest detailed deployment instructions.
- **CHANGELOG.md**: Detailed history of all changes.
- **ROADMAP.md**: Major long-term structural plans.
- **TODO.md**: Individual features, bug fixes, and short-term fine details.
- **HANDOFF.md**: Session history and memories for the next agent.

## 🚫 Strictly Prohibited Actions

- Do NOT \`taskkill\` node processes.
- Do NOT sacrifice audio quality for speed.
- Do NOT stop until the task is complete.

---
*Magnificent! Extraordinary! Insanely Great!*
