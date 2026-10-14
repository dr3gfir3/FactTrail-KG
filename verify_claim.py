import sys
from src.pipeline import verify

def main():
    if len(sys.argv) > 1:
        claim_text = " ".join(sys.argv[1:])
    else:
        print("=== Fact-Checking Pipeline (Wiki-First) ===")
        claim_text = input("Inserisci il claim da verificare (in inglese): ").strip()
        if not claim_text:
            print("Il claim non pu\u00f2 essere vuoto.")
            return

    result = verify(claim_text)
    probs = result["probs"]

    verdict_colors = {
        "SUPPORTED": "\033[92m",
        "REFUTED": "\033[91m",
        "NOT ENOUGH INFO": "\033[93m",
    }
    color = verdict_colors.get(result["verdict"], "\033[93m")
    reset = "\033[0m"

    print("\n" + "="*50)
    print("RISULTATI DELLA VERIFICA")
    print("="*50)
    print(f"Claim: {claim_text}")
    print(f"Entit\u00e0: {result['entities']}")
    print(f"Fonte: {'Wiki Locale' if result['source'] == 'wiki' else 'Nessuna evidenza'}")
    print(f"Pagine Wiki aggiunte: {result['wiki_pages_added']}")
    print(f"Trivelle KG totali: {result['wiki_triples']}")
    print(f"Premise: {result['premise'][:300]}...")
    print(f"Verdetto: {color}{result['verdict']}{reset}")
    print(f"Confidenza: {result['confidence']*100:.1f}%")
    print(f"  Supportato:      {probs.get('SUPPORTED', 0)*100:.1f}%")
    print(f"  Confutato:       {probs.get('REFUTED', 0)*100:.1f}%")
    print(f"  Info Insuff.:    {probs.get('NOT ENOUGH INFO', 0)*100:.1f}%")
    print(f"Tempi: {result['stages']}")
    print("="*50)

if __name__ == "__main__":
    main()
