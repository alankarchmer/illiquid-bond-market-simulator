# Quick Start: How to Win at the Bond Trading Game

## TL;DR - The Formula for Success

When you see an RFQ, calculate your quote like this:

```
1. Start with the "Naive mid estimate" shown
2. Add/subtract based on:
   - Liquidity: LOW liquidity → WIDER spread
   - Position: Already have inventory → quote away from more
   - Client type: Shows in final summary, learn who's toxic

Formula:
  Spread = 0.25 + (1 - liquidity) * 1.0
  
  If client BUYING from you:  quote = mid + spread
  If client SELLING to you:   quote = mid - spread
```

## Real Examples from Your Next Game

### Example 1: Liquid Bond, Neutral Position
```
RFQ: Client RM01 wants to BUY 2.0 units
Bond: BOND015 (TechGiant, IG, A)
Liquidity: 0.75 (fairly liquid)
Naive mid: 101.50
Your position: 0.00
```

**Your calculation:**
```
Spread = 0.25 + (1 - 0.75) * 1.0 = 0.25 + 0.25 = 0.50
Client buying → you're selling → quote above mid
Quote: 101.50 + 0.25 = 101.75
```

Type: **`101.75`**

Expected outcome: 60% chance of fill, earn ~0.25 profit

---

### Example 2: Illiquid Bond
```
RFQ: Client HF01 wants to SELL 1.5 units
Bond: BOND042 (Mining Co, HY, B)
Liquidity: 0.18 (very illiquid!)
Naive mid: 98.75
Your position: 0.00
```

**Your calculation:**
```
Spread = 0.25 + (1 - 0.18) * 1.0 = 0.25 + 0.82 = 1.07
Client selling → you're buying → quote below mid
Quote: 98.75 - 0.50 = 98.25
```

Type: **`98.25`**

Expected outcome: 30% chance of fill (wide spread), but big profit margin

---

### Example 3: Managing Inventory
```
RFQ: Client NO02 wants to SELL 2.0 units  
Bond: BOND020 (already own 4.5 units!)
Liquidity: 0.45
Naive mid: 100.25
Your position: +4.5 (LONG - you own a lot!)
```

**Your calculation:**
```
Base spread = 0.25 + (1 - 0.45) * 1.0 = 0.80
Inventory adjustment = -0.30 (discourage buying more)
Client selling → you're buying → quote below mid
Quote: 100.25 - 0.80 = 99.45
```

Type: **`99.45`** (or even lower like `99.00` to really discourage)

Expected outcome: Low fill chance (intentional - you don't want more)

---

### Example 4: You're Short, Want to Cover
```
RFQ: Client RM02 wants to BUY 1.0 units
Bond: BOND033 (you're short -3.2 units)
Liquidity: 0.52  
Naive mid: 99.80
Your position: -3.2 (SHORT - you owe it!)
```

**Your calculation:**
```
Base spread = 0.25 + (1 - 0.52) * 1.0 = 0.73
Inventory adjustment = TIGHTEN (you WANT to sell to reduce short)
Client buying → you're selling → quote above mid BUT TIGHT
Quote: 99.80 + 0.25 = 100.05
```

Type: **`100.05`** (tight to encourage fill)

Expected outcome: High fill chance - helps you reduce risk!

---

## Decision Tree

```
START
  ↓
Is liquidity < 0.3?
  ├─ YES → Use WIDE spread (1.0+ points)
  └─ NO → Use NORMAL spread (0.3-0.6 points)
  ↓
Do you have a position in this bond?
  ├─ Same direction as trade → Make spread EVEN WIDER
  ├─ Opposite direction → Keep spread tight (helps you)
  └─ No position → Use base spread
  ↓
Is this a new client?
  ├─ YES → Use normal spread
  └─ NO → Check their avg edge from previous games
       ├─ High positive edge (they win) → Add 50% to spread
       └─ Negative edge (you win) → Keep spread normal
  ↓
QUOTE YOUR PRICE
```

## Common Scenarios Cheat Sheet

| Situation | What to Do | Why |
|-----------|------------|-----|
| Liquidity 0.8, no position | Mid ± 0.30 | Easy to trade, tight spread |
| Liquidity 0.2, no position | Mid ± 1.00 | Hard to unwind, need compensation |
| Long 5 units, client selling more | Quote LOW (mid - 1.5) | Discourage - you have enough |
| Short 3 units, client wants to buy | Quote TIGHT (mid + 0.20) | Encourage - helps you cover |
| Fisher/hedge fund client | Add 0.50 to spread | They might know something |
| Noise trader (fill every RFQ) | Widen by 50% | Take advantage of irrationality |

## What Good Performance Looks Like

After 30 steps, you want:
- **P&L: +2 to +5** (positive is key!)
- **Fill Ratio: 30-50%** (too high means quotes too tight)
- **Inventory Risk: <20** (not over-exposed)
- **Average Edge: Negative** (clients lose, you win)

## Practice Commands

Try different scenarios:

```bash
# Easy mode (fewer bonds, more time)
python3 cli.py --game --steps 30 --bonds 20

# Standard game  
python3 cli.py --game --steps 30 --bonds 50

# Hard mode (more bonds, less time)
python3 cli.py --game --steps 50 --bonds 100

# Same random scenario (to practice)
python3 cli.py --game --steps 30 --seed 42
```

## Pro Tips

1. **Write down patterns**: Keep a notepad of which clients are toxic
2. **Start conservative**: Use wider spreads initially
3. **Adjust as you learn**: Tighten spreads on clients with negative edge
4. **Inventory is risk**: Try to stay under 15 total inventory
5. **Don't pass**: Always quote a real price, "pass" causes bugs

## When You're Ready to Code Your Own Strategy

Check out `examples_ai_agent.py` for templates:
- **MLQuotingStrategy**: For machine learning approaches
- **RLQuotingAgent**: For reinforcement learning
- **LLMQuotingAgent**: For LLM-based reasoning

The simulation now correctly tracks P&L, so you can train agents and see real results!

---

**Good luck! May your P&L be positive and your spreads be wide! 📈💰**
