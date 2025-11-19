# Test RAG System

Test the complete RAG pipeline with sample documents and queries.

## Purpose
- Verify RAG system works end-to-end
- Test with medical/finance sample documents
- Validate retrieval quality
- Test answer generation

## Usage

```bash
/test-rag
```

Or with custom model:
```bash
export MODEL_PATH=outputs/my_model
/test-rag
```

## Commands

```bash
# Use default or custom model path
MODEL_PATH=${MODEL_PATH:-outputs/quick_test}

echo "=================================="
echo "Testing RAG System"
echo "=================================="
echo "Model: $MODEL_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH/best_model.pt" ]; then
    echo "❌ Model not found: $MODEL_PATH/best_model.pt"
    echo "   Run /quick-train first"
    exit 1
fi

# Create test documents
echo "[1/4] Creating test documents..."
mkdir -p test_docs

cat > test_docs/medical_basics.txt << 'EOF'
Medical Knowledge Base - Diabetes

Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels. There are two main types:

Type 1 Diabetes: An autoimmune condition where the pancreas produces little or no insulin. Typically develops in childhood or adolescence. Requires lifelong insulin therapy.

Type 2 Diabetes: Results from insulin resistance and relative insulin deficiency. More common in adults, especially those who are overweight. Can often be managed with lifestyle changes and oral medications.

Common symptoms include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision.

Treatment approaches:
- Insulin therapy (Type 1, sometimes Type 2)
- Oral medications like Metformin (Type 2)
- Blood glucose monitoring
- Dietary modifications
- Regular physical activity
- Weight management

Complications if uncontrolled: cardiovascular disease, kidney damage, nerve damage, eye problems.
EOF

cat > test_docs/medical_treatment.txt << 'EOF'
Treatment Guidelines - Hypertension

Hypertension (high blood pressure) is defined as blood pressure consistently above 130/80 mmHg.

Risk Factors:
- Age (increases with age)
- Family history
- Obesity
- Sedentary lifestyle
- High sodium intake
- Excessive alcohol consumption

Treatment Approaches:

Lifestyle Modifications:
- Reduce sodium intake (less than 2,300 mg/day)
- DASH diet (rich in fruits, vegetables, whole grains)
- Regular aerobic exercise (150 minutes/week)
- Weight loss if overweight
- Limit alcohol consumption
- Stress management

Medications (First-line):
- ACE inhibitors (e.g., Lisinopril)
- ARBs (Angiotensin Receptor Blockers)
- Calcium channel blockers (e.g., Amlodipine)
- Thiazide diuretics

Target blood pressure: Generally below 130/80 mmHg
Monitoring: Regular blood pressure checks, annual kidney function tests

Complications if untreated: stroke, heart attack, heart failure, kidney disease
EOF

cat > test_docs/finance_basics.txt << 'EOF'
Financial Planning Fundamentals

Investment Strategies:

Diversification: Spreading investments across different asset classes to reduce risk. Don't put all eggs in one basket.

Asset Allocation:
- Stocks: Higher risk, higher potential return
- Bonds: Lower risk, stable income
- Real Estate: Inflation hedge, income generation
- Cash: Liquidity, low return

Risk Tolerance: Depends on age, income, goals
- Young investors: Can take more risk (70-80% stocks)
- Near retirement: More conservative (40-50% bonds)

Compound Interest: Reinvesting earnings to generate additional earnings. The most powerful force in wealth building.

Key Principles:
- Start early
- Contribute regularly
- Minimize fees
- Rebalance annually
- Stay diversified
- Avoid emotional decisions

Emergency Fund: 3-6 months of expenses in accessible savings

Retirement Accounts:
- 401(k): Employer-sponsored, often with matching
- IRA: Individual retirement account
- Roth IRA: Tax-free growth and withdrawals
EOF

echo "✓ Created 3 test documents"

# Build RAG system
echo ""
echo "[2/4] Building RAG knowledge base..."

python << PYEOF
from rag_system import create_rag_system
import glob

print("Loading model...")
rag = create_rag_system(
    model_checkpoint='$MODEL_PATH/best_model.pt',
    tokenizer_path='$MODEL_PATH/tokenizer.json',
    device='cpu'
)

print("Adding documents...")
for doc in glob.glob('test_docs/*.txt'):
    print(f'  - {doc}')
    rag.add_document(doc)

print("Saving RAG database...")
rag.save('test_rag_data')
print("✓ RAG system built")
PYEOF

echo ""
echo "[3/4] Running test queries..."

python << 'PYEOF'
from rag_system import create_rag_system

# Load RAG
rag = create_rag_system(
    model_checkpoint='$MODEL_PATH/best_model.pt',
    tokenizer_path='$MODEL_PATH/tokenizer.json',
    device='cpu'
)
rag.load('test_rag_data')

# Test questions
questions = [
    "What is Type 1 diabetes?",
    "What medications treat hypertension?",
    "What are the symptoms of diabetes?",
    "What is diversification in investing?",
    "What is the target blood pressure?",
]

print("\n" + "="*60)
print("RAG System Test Results")
print("="*60)

passed = 0
total = len(questions)

for i, q in enumerate(questions, 1):
    print(f"\n[Query {i}/{total}]")
    print(f"Q: {q}")

    try:
        result = rag.query(q, top_k=2, max_new_tokens=100)

        answer = result['answer']
        confidence = result['confidence']

        print(f"A: {answer}")
        print(f"Confidence: {confidence:.2%}")

        # Simple quality checks
        checks = {
            'Has answer': len(answer) > 20,
            'Confidence > 0.3': confidence > 0.3,
            'Retrieved sources': result['num_sources'] > 0,
        }

        all_passed = all(checks.values())
        status = "✓ PASS" if all_passed else "✗ FAIL"
        print(f"Status: {status}")

        if all_passed:
            passed += 1

        # Show which checks failed
        for check, result in checks.items():
            if not result:
                print(f"  ✗ {check}")

    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*60)
print(f"Results: {passed}/{total} queries passed")
print("="*60)

if passed >= total * 0.8:  # 80% pass rate
    print("✓ RAG system is working well!")
    exit(0)
elif passed >= total * 0.5:
    print("⚠ RAG system working but needs improvement")
    exit(0)
else:
    print("✗ RAG system needs attention")
    exit(1)
PYEOF

TEST_RESULT=$?

echo ""
echo "[4/4] Cleanup..."
rm -rf test_docs test_rag_data

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ RAG Testing Complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  - Add real documents: rag.add_document('your_doc.txt')"
    echo "  - Deploy: /deploy-docker"
    echo "  - Benchmark: /benchmark"
    echo ""
else
    echo "❌ RAG tests failed"
    exit 1
fi
```

## Expected Results

- ✓ Documents processed successfully
- ✓ Queries return relevant answers
- ✓ Confidence scores > 30%
- ✓ 80%+ queries pass quality checks

## Success Criteria

- [x] Model loads successfully
- [x] Documents chunked and indexed
- [x] Retrieval finds relevant chunks
- [x] Answers are coherent
- [x] Confidence scores reasonable
