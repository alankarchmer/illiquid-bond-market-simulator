# How to Make Money in the Bond Market Simulator

## The Core Problem: Why Your P&L is 0.00

### What's Happening
Your P&L shows 0.00 because of a mark-to-market bug:
- When you "pass" on an RFQ, the system quotes extreme prices (999.00 or 0.01)
- Noise traders sometimes fill these extreme quotes
- The system marks your positions using the **last traded price**
- Since you traded at 999.00 and mark at 999.00, your P&L = 0

### The Bug
Look at `illiquid_market_sim/portfolio.py` lines 174-176:
```python
current_price = bond.get_last_traded_price()
if current_price is None:
    current_price = bond.get_true_fair_price()
```

This uses your own trade price to mark your position! This is incorrect for market making.

## How to Actually Make Money

### 1. **Stop Passing on RFQs**
When you type "pass", you're quoting 999.00 or 0.01, which creates this bug. Instead:
- **Always quote a real price**
- Even if uncertain, quote around 100 with a wide spread

### 2. **Understand Bid-Ask Spreads**
The dealer makes money by earning the spread:

**Example:**
- Bond true value: 100.00
- You quote:
  - **Bid (when client sells TO you)**: 99.50 (you buy low)
  - **Offer (when client buys FROM you)**: 100.50 (you sell high)
- If you complete both trades, you make: 100.50 - 99.50 = **1.00 profit**

### 3. **Pricing Strategy**

When you get an RFQ:

```
RFQ shows:
- Naive mid estimate: 100.00
- Client wants to BUY (you sell)
- Size: 2.0
- Current position: 0
```

**Your quote should be:**
```
Base mid = 100.00
Spread = 0.50 points (50 bps)
Client buying → You selling → Quote ABOVE mid
Final price = 100.00 + 0.25 = 100.25
```

### 4. **Key Factors to Consider**

#### A. **Liquidity**
- Liquid bonds (0.7-0.8): Use narrow spreads (20-30 bps)
- Illiquid bonds (0.1-0.3): Use WIDE spreads (100-200 bps)

#### B. **Inventory Management**
- Already LONG the bond? Quote LOWER when buying more (discourage)
- Already SHORT the bond? Quote HIGHER when selling more (discourage)
- Neutral position? Quote normally

#### C. **Client Type** (shown in final summary)
- **Noise traders**: Quote wide - they'll trade at bad prices
- **Fisher (informed) traders**: Quote VERY wide - they know something you don't
- **Real money**: Normal spreads
- **Hedge funds**: Slightly wider - they might be informed

### 5. **Example Game Play**

Let's say you see:
```
RFQ: Client NO01 wants to BUY 1.5 units
Bond: BOND028 (Retail, HY, BB)
Maturity: 5.2y, Liquidity: 0.25
Naive mid estimate: 100.00
Current position: 0.00
```

**Your thinking:**
1. Low liquidity (0.25) → use WIDE spread
2. Client is NO01 (noise trader from your output) → they're dumb, go wider
3. HY bond → riskier, wider spread
4. No position → neutral

**Your quote calculation:**
```
Mid = 100.00
Base spread = 1.00 (100 bps due to low liquidity)
Client buying → add spread to mid
Quote: 100.50
```

Type: `100.50`

### 6. **Common Mistakes**

❌ **Typing "pass" too often** → Creates the 999.00 bug
❌ **Quoting too tight** → Low fill ratio (8.7% like yours)
❌ **Ignoring inventory** → Building large risky positions
❌ **Same spread for all bonds** → Liquid and illiquid need different treatment

✅ **Quote real prices** → Between 98-102 typically
✅ **Wider on illiquid bonds** → 0.5-2.0 point spreads
✅ **Adjust for inventory** → Skew quotes if you have position
✅ **Quote wider to informed clients** → Protect yourself

## Quick Reference Table

| Situation | Your Action | Example Quote |
|-----------|-------------|---------------|
| Client BUYING, liquid bond (0.7) | Quote slightly above mid | Mid 100 → Quote **100.10** |
| Client SELLING, liquid bond (0.7) | Quote slightly below mid | Mid 100 → Quote **99.90** |
| Client BUYING, illiquid bond (0.2) | Quote well above mid | Mid 100 → Quote **100.75** |
| Client SELLING, illiquid bond (0.2) | Quote well below mid | Mid 100 → Quote **99.25** |
| Client BUYING, you're already SHORT | Quote normally (helps you) | Mid 100 → Quote **100.10** |
| Client BUYING, you're already LONG 5 | Quote very high (avoid more) | Mid 100 → Quote **101.50** |
| Suspicious informed client | Add 50% to spread | Normal 100.20 → Quote **100.30** |

## The Fix for the Bug

If you want to fix the mark-to-market bug, edit `illiquid_market_sim/portfolio.py` around line 174:

**Current (buggy):**
```python
current_price = bond.get_last_traded_price()
if current_price is None:
    current_price = bond.get_true_fair_price()
```

**Better approach:**
```python
# Don't use last traded if it's an obvious outlier
last_price = bond.get_last_traded_price()
true_fair = bond.get_true_fair_price()

if last_price is None:
    current_price = true_fair
elif abs(last_price - true_fair) > 50:  # Obvious outlier
    current_price = true_fair
else:
    # Blend last price with fair value
    current_price = 0.7 * last_price + 0.3 * true_fair
```

## Pro Tips

1. **Start conservative**: Quote 0.50-1.00 point spreads initially
2. **Tighten as you learn**: Once profitable, try tighter spreads for more volume
3. **Track your edge**: Check the "Avg Edge" column - negative is good for you!
4. **Avoid toxic clients**: If someone has high fill ratio + large negative edge, quote wide to them
5. **Inventory limits**: Try to keep total inventory under 20

## Try It Again!

Run the game mode again:
```bash
python cli.py --game --steps 30 --seed 42
```

This time:
- **Always quote real prices** (not "pass")
- Start with mid ± 0.50 for most bonds
- Go wider (± 1.00) on illiquid bonds
- Adjust for your position

Good luck! 🎲📈
