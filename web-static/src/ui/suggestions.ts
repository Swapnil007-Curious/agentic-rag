// Three suggested questions per collection. Each was checked against the collection's
// passages.json (the section that answers it is noted) before it was chosen.

export const SUGGESTIONS: Record<string, string[]> = {
  cs_ai: [
    "What does git rebase do, and when is it dangerous?", // Pro Git > Git Branching > Rebasing, The Perils of Rebasing
    "How does a convolutional layer differ from a fully connected layer?", // D2L ch. 7.1 From Fully Connected Layers to Convolutions
    "What is the difference between a list and a tuple in Python?", // Python Tutorial 5.3 Tuples and Sequences
  ],
  consumer_law: [
    "Who can file a complaint before a District Commission?", // CPA 2019 s. 35
    "What is the time limit for filing a consumer complaint?", // CPA 2019 s. 69 Limitation period
    "What must an e-commerce entity do under the 2020 Rules?", // E-Commerce Rules 2020 r. 4, 5
  ],
  indian_law: [
    "What does Article 21 of the Constitution protect?", // Constitution Part III Art. 21
    "Who is liable to register under the CGST Act, 2017?", // CGST s. 22
    "What is the punishment for murder under the Bharatiya Nyaya Sanhita?", // BNS 2023 s. 103
  ],
  human_biology: [
    "How does a muscle fiber contract?", // A&P 2e 10.3 Sliding Filament Model
    "What are the chambers and valves of the heart?", // A&P 2e ch. 19 Chambers and Circulation
    "How does the body keep blood pH within its normal range?", // A&P 2e ch. 26 Acid-Base Balance
  ],
  physics_chemistry: [
    "What does Archimedes' principle state?", // College Physics 2e 11.7
    "How does the ideal gas law relate pressure, volume and temperature?", // College Physics 2e 13.3
    "What is the difference between an SN1 and an SN2 reaction?", // Organic Chemistry 11.2–11.4
  ],
};

export const GENERAL_SUGGESTIONS = ["Explain big-O notation with an example", "What is the difference between a process and a thread?", "Derive the kinematic equations"];

export function suggestionsFor(collection: string | null): string[] {
  return (collection && SUGGESTIONS[collection]) || GENERAL_SUGGESTIONS;
}
