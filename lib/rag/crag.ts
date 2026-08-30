import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";

const RELEVANCE_PROMPT = `You are a relevance judge for a RAG system.
Given a user query and a retrieved document chunk, determine if the chunk is RELEVANT or IRRELEVANT to answering the query.

Respond with exactly one word: RELEVANT or IRRELEVANT.`;

export interface RelevanceResult {
  relevant: Document[];
  irrelevant: Document[];
}

export async function filterRelevant(
  query: string,
  docs: Document[]
): Promise<RelevanceResult> {
  if (docs.length <= 3) {
    return { relevant: docs, irrelevant: [] };
  }

  const model = new ChatMistralAI({
    modelName: process.env.MISTRAL_MODEL || "mistral-small-latest",
    apiKey: process.env.MISTRAL_API_KEY,
    temperature: 0,
  });

  const relevant: Document[] = [];
  const irrelevant: Document[] = [];

  const BATCH_SIZE = 5;
  for (let i = 0; i < docs.length; i += BATCH_SIZE) {
    const batch = docs.slice(i, i + BATCH_SIZE);
    const judgements = await Promise.all(
      batch.map(async (doc) => {
        try {
          const response = await model.invoke([
            new SystemMessage(RELEVANCE_PROMPT),
            new HumanMessage(
              `Query: ${query}\n\nDocument chunk:\n${doc.pageContent.slice(0, 1500)}`
            ),
          ]);
          return String(response.content).trim().toUpperCase().startsWith("RELEVANT");
        } catch {
          return true;
        }
      })
    );

    batch.forEach((doc, idx) => {
      if (judgements[idx]) {
        relevant.push(doc);
      } else {
        irrelevant.push(doc);
      }
    });
  }

  if (relevant.length === 0) {
    return { relevant: docs.slice(0, 2), irrelevant: docs.slice(2) };
  }

  return { relevant, irrelevant };
}

export function needsReRetrieval(relevant: number, total: number): boolean {
  if (total === 0) return true;
  return relevant / total < 0.3;
}
