import { callGemini } from "./gemini";

const HYDE_SYSTEM_PROMPT = `You are a query rephrasing engine for portfolio search.
Given a user's question about Mohamed Maamar's portfolio, generate 3 semantically similar rephrasings that use alternative keywords and phrasing.
- Generate 6 different ways to ask the same question
- Use synonyms and related terminology
- Focus on the core intent, not technical jargon
- Do NOT invent technologies or frameworks not mentioned in the original query
- Keep them grounded and realistic (each under 100 words)
- Return ONLY the 3 rephrased questions, one per line, no numbering or explanations`;

export async function generateQueryVariations(query: string): Promise<string[]> {
  try {
    const responseText = await callGemini(query, HYDE_SYSTEM_PROMPT, 0.3);
    const variations = responseText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 5);

    return [query, ...variations];
  } catch {
    return [query];
  }
}
