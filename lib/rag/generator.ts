import { ChatMistralAI } from "@langchain/mistralai";
import { CohereRerank } from "@langchain/cohere";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import type { RagResult } from "./types";

const GREETINGS_RE =
  /^(hey|hi|hello|yo+|yoo+|sup|what'?s up|howdy|greetings|salut|bonjour|hola|ciao|hei|مرحبا|السلام عليكم)\b/i;

const SYSTEM_PROMPT = `You are an expert technical portfolio assistant representing Mohamed Maamar (Med).

<context>
{numberedContext}
</context>

RULES & GUIDELINES:
1. COMPREHENSIVENESS: Thoroughly analyze the provided context. When asked about a broad topic (like "agents" or "projects"), you MUST synthesize all relevant information across the entire context. Do not stop after finding the first example. Mention ALL related projects, tools, and technical details present in the context. CRITICAL: If asked to list items (like agents), you MUST list every single one found across all documents. Do not exclude an item just because it belongs to a different project or ecosystem than the others. If a project or ecosystem contains multiple distinct agents (e.g., Agent 1, Agent 2), you MUST enumerate each distinct agent individually rather than just grouping them under the ecosystem name.
2. GROUNDING: Answer ONLY using the provided context. DO NOT infer or hallucinate agents/projects that are not explicitly labeled as such in the text (e.g., do not call VTON an "agent" unless the text explicitly does). If the context does not contain the answer, explicitly state: "I don't have that detail — reach Med directly at almohamedmaamar@gmail.com."
3. FORMATTING: Do NOT use any markdown formatting. No stars (**), no hash symbols (#), no dashes (-). Just plain text with line breaks. Do not output raw XML tags like <context> to the user.
4. TONE: Professional, highly technical, and confident.`;

const GREETING_RESPONSE =
  "Hey! I'm Med's portfolio assistant. Ask me anything about his engineering experience, AI projects, or technical skills.";

export async function rerank(
  docs: Document[],
  query: string,
  topN = 10
): Promise<Document[]> {
  if (docs.length === 0) return [];
  const reranker = new CohereRerank({
    apiKey: process.env.COHERE_API_KEY,
    model: "rerank-english-v3.0",
    topN,
  });
  return reranker.compressDocuments(docs, query);
}

export function isSmallTalk(query: string): boolean {
  return GREETINGS_RE.test(query.trim());
}

interface ChatMessage { role: string; content: string; }

export async function generate(
  docs: Document[],
  query: string,
  history?: ChatMessage[]
): Promise<RagResult> {
  const numberedContext = docs
    .map((d, i) => {
      const clean = d.pageContent.replace(/^\[Document:[^\]]*\]\s*/m, "");
      return `[Source ${i + 1}: ${d.metadata?.title || 'Document'} | Section: ${d.metadata?.section || 'General'}]\n${clean}`;
    })
    .join("\n\n---\n\n");

  const conversationHistory = history && history.length > 0
    ? history.map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`).join("\n")
    : "";

  const fullPrompt = conversationHistory
    ? `Previous conversation:\n${conversationHistory}\n\nCurrent question: ${query}`
    : query;

  const citations = Array.from(
    new Set(
      docs
        .map((d) => (d.metadata?.sourceFile || d.metadata?.title) as string | undefined)
        .filter((s): s is string => Boolean(s))
    )
  );

  const model = new ChatMistralAI({
    modelName: process.env.MISTRAL_MODEL || "mistral-small-latest",
    apiKey: process.env.MISTRAL_API_KEY,
    temperature: 0.1,
  });

  const response = await model.invoke([
    new SystemMessage(SYSTEM_PROMPT.replace("{numberedContext}", numberedContext)),
    new HumanMessage(fullPrompt),
  ]);

  return {
    content: formatResponse(String(response.content)),
    citations,
  };
}

function formatResponse(text: string): string {
  let t = text.trim();

  t = t.replace(/#{1,6}\s+/g, "");
  t = t.replace(/\*\*(.+?)\*\*/g, "$1");
  t = t.replace(/\*(.+?)\*/g, "$1");
  t = t.replace(/^[-•]\s*/gm, "• ");
  t = t.replace(/`(.+?)`/g, "$1");
  t = t.replace(/~~(.+?)~~/g, "$1");
  t = t.replace(/\[(\d+(?:,\s*\d+)*)\]/g, "");
  t = t.replace(/\[(.+?)\]\(.+?\)/g, "$1");
  t = t.replace(/\n{3,}/g, "\n\n");
  t = t.replace(/\[\s*\]/g, "");

  const lines = t.split("\n");
  const formatted: string[] = [];
  let inList = false;

  for (const line of lines) {
    const trimmed = line.trimEnd();
    const isBullet = /^[•\-\*]\s/.test(trimmed);
    const isEmpty = trimmed === "";

    if (isBullet) {
      if (!inList && formatted.length > 0 && formatted[formatted.length - 1] !== "") {
        formatted.push("");
      }
      inList = true;
      formatted.push(trimmed);
    } else if (isEmpty) {
      inList = false;
      formatted.push("");
    } else {
      inList = false;
      formatted.push(trimmed);
    }
  }

  return formatted.join("\n");
}

export { GREETING_RESPONSE };
