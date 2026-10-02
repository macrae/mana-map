package forge.ai.ability;

import forge.ai.*;
import forge.game.Game;
import forge.game.GameObject;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.combat.Combat;
import forge.game.cost.Cost;
import forge.game.cost.CostPart;
import forge.game.cost.CostPayLife;
import forge.game.cost.CostRemoveCounter;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PumpAllAi extends PumpAiBase {

    /* (non-Javadoc)
     * @see forge.card.abilityfactory.SpellAiLogic#canPlayAI(forge.game.player.Player, java.util.Map, forge.card.spellability.SpellAbility)
     */
    @Override
    protected AiAbilityDecision checkApiLogic(final Player ai, final SpellAbility sa) {
        final Card source = sa.getHostCard();
        final Game game = ai.getGame();
        final Combat combat = game.getCombat();
        final Cost abCost = sa.getPayCosts();
        final String logic = sa.getParamOrDefault("AILogic", "");

        if (logic.equals("UntapCombatTrick")) {
            PhaseHandler ph = ai.getGame().getPhaseHandler();
            if (!(ph.is(PhaseType.COMBAT_DECLARE_BLOCKERS, ai)
                    || (!ph.getPlayerTurn().equals(ai) && ph.is(PhaseType.COMBAT_DECLARE_ATTACKERS)))) {
                return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
            }
        }

        if (abCost != null && source.hasSVar("AIPreference")) {
            if (!ComputerUtilCost.checkSacrificeCost(ai, abCost, source, sa, true)) {
                return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
            }
        }
        
        final Player opp = ai.getStrongestOpponent();

        if (sa.usesTargeting()) {
            if (sa.canTarget(opp) && sa.isCurse()) {
                sa.resetTargets();
                sa.getTargets().add(opp);
                return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
            }

            if (sa.canTarget(ai) && !sa.isCurse()) {
                sa.resetTargets();
                sa.getTargets().add(ai);
                return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
            }
        }

        // MANA MAP PATCH (2026-10-01): A SWEEPER WHOSE X IS ITS OWN COST READS AS -0/-0.
        //
        // Toxic Deluge is `SP$ PumpAll | Cost$ 2 B PayLife<X> | NumAtt$ -X | NumDef$ -X` with
        // `SVar:X:Count$xPaid`, and `xPaid` is 0 until the cost is PAID — so `defense` below
        // computed to 0, every creature "survived" the filter, and the curse branch (which is
        // the only branch that can ever want a -X/-X) was never worth entering. Measured: in a
        // 200-game pod run Toxic Deluge was DRAWN 28 times by our seat, castable and uncast on
        // 115 own turns, cast 0, discarded 0; in an 8-game shell, 0 casts over 22 castable
        // turns. The same defect `PumpAi.java` carries for Vish Kal's -X/-X, on another API.
        //
        // SECOND HALF OF THE SAME DEFECT: the script also has no `IsCurse$`, and `isCurse()` is
        // `hasParam("IsCurse")`, so even a correctly priced -X/-X would have been read as a
        // PUMP for our own creatures. A PumpAll whose NumDef is negative is a sweeper whatever
        // the script says, so the curse test below accepts that too — a fact about the effect,
        // not a guess about the author's intent.
        final int rawPower = AbilityUtils.calculateAmount(source, sa.getParam("NumAtt"), sa);
        final int rawDefense = AbilityUtils.calculateAmount(source, sa.getParam("NumDef"), sa);
        final boolean attNegX = String.valueOf(sa.getParam("NumAtt")).startsWith("-");
        final boolean defNegX = String.valueOf(sa.getParam("NumDef")).startsWith("-");
        final boolean negativeX = rawPower == 0 && rawDefense == 0 && (attNegX || defNegX)
                && "Count$xPaid".equals(sa.getSVar("X"));
        int chosenX = 0;
        if (negativeX) {
            chosenX = chooseSweeperX(ai, sa);
            if (chosenX <= 0) {
                return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
            }
            sa.getRootAbility().setXManaCostPaid(chosenX);
        }
        final int power = (negativeX && attNegX) ? -chosenX : rawPower;
        final int defense = (negativeX && defNegX) ? -chosenX : rawDefense;
        final boolean curse = sa.isCurse() || defense < 0;
        final List<String> keywords = sa.hasParam("KW") ? Arrays.asList(sa.getParam("KW").split(" & ")) : new ArrayList<>();
        final PhaseType phase = game.getPhaseHandler().getPhase();

        final String valid = sa.getParamOrDefault("ValidCards", "");

        CardCollection comp = CardLists.getValidCards(ai.getCardsIn(ZoneType.Battlefield), valid, source.getController(), source, sa);
        CardCollection human = CardLists.getValidCards(opp.getCardsIn(ZoneType.Battlefield), valid, source.getController(), source, sa);

        if (curse) {
            if (defense < 0) { // try to destroy creatures
                comp = CardLists.filter(comp, c -> {
                    if (c.getNetToughness() <= -defense) {
                        return true; // can kill indestructible creatures
                    }
                    return ComputerUtilCombat.getDamageToKill(c, false) <= -defense && !c.hasKeyword(Keyword.INDESTRUCTIBLE);
                }); // leaves all creatures that will be destroyed
                human = CardLists.filter(human, c -> {
                    if (c.getNetToughness() <= -defense) {
                        return true; // can kill indestructible creatures
                    }
                    return ComputerUtilCombat.getDamageToKill(c, false) <= -defense && !c.hasKeyword(Keyword.INDESTRUCTIBLE);
                }); // leaves all creatures that will be destroyed
            } // -X/-X end
            else if (power < 0) { // -X/-0
                if (phase.isAfter(PhaseType.COMBAT_DECLARE_BLOCKERS)
                        || phase.isBefore(PhaseType.COMBAT_DECLARE_ATTACKERS)
                        || game.getPhaseHandler().isPlayerTurn(sa.getActivatingPlayer())
                        || game.getReplacementHandler().isPreventCombatDamageThisTurn()) {
                    return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
                }
                int totalPower = 0;
                for (Card c : human) {
                    if (combat == null || !combat.isAttacking(c)) {
                        continue;
                    }
                    totalPower += Math.min(c.getNetPower(), power * -1);
                    if (phase == PhaseType.COMBAT_DECLARE_BLOCKERS && combat.isUnblocked(c)) {
                        if (ComputerUtilCombat.lifeInDanger(sa.getActivatingPlayer(), combat)) {
                            return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
                        }
                        totalPower += Math.min(c.getNetPower(), power * -1);
                    }
                    if (totalPower >= power * -2) {
                        return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
                    }
                }
                return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
            } // -X/-0 end
            
            if (comp.isEmpty() && ComputerUtil.activateForCost(sa, ai)) {
            	return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
            }

            // evaluate both lists and pass only if human creatures are more valuable
            boolean result = (ComputerUtilCard.evaluateCreatureList(comp) + 200) < ComputerUtilCard.evaluateCreatureList(human);
            return result ? new AiAbilityDecision(100, AiPlayDecision.WillPlay) : new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        } // end Curse

        if (!game.getStack().isEmpty()) {
            boolean result = pumpAgainstRemoval(ai, sa, comp);
            return result ? new AiAbilityDecision(100, AiPlayDecision.WillPlay) : new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }

        boolean result = ai.getCreaturesInPlay().anyMatch(c -> c.isValid(valid, source.getController(), source, sa)
                && ComputerUtilCard.shouldPumpCard(ai, sa, c, defense, power, keywords));
        return result ? new AiAbilityDecision(100, AiPlayDecision.WillPlay) : new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
    }

    // MANA MAP PATCH (2026-10-01): how big a sweeper to cast.
    //
    // Walk every X the cost can actually pay and keep the one that clears the most VALUE off
    // the opponents' boards net of our own, by the engine's own `evaluateCreatureList`. The
    // margin must be positive — a symmetric sweeper that trades evenly is a card for nothing —
    // and a life cost may not take us below a reserve, because `PayLife<X>` on an empty board
    // would otherwise happily pay our last points. Capped at 12 so the loop is bounded on a
    // card with no other limit.
    private static final int SWEEPER_X_CAP = 12;
    private static final int LIFE_RESERVE = 8;
    private static final int MARGIN = 100;

    private int chooseSweeperX(final Player ai, final SpellAbility sa) {
        final Card source = sa.getHostCard();
        final String valid = sa.getParamOrDefault("ValidCards", "");
        int payLifeCap = Integer.MAX_VALUE;
        final Cost cost = sa.getPayCosts();
        if (cost != null) {
            for (final CostPart part : cost.getCostParts()) {
                if (part instanceof CostPayLife && "X".equals(part.getAmount())) {
                    payLifeCap = Math.max(0, ai.getLife() - LIFE_RESERVE);
                }
            }
        }
        final int cap = Math.min(SWEEPER_X_CAP, payLifeCap);
        int bestX = 0;
        int bestMargin = 0;
        for (int x = 1; x <= cap; x++) {
            final int lethal = x;
            final CardCollection oursDead = CardLists.filter(
                    CardLists.getValidCards(ai.getCardsIn(ZoneType.Battlefield), valid, source.getController(), source, sa),
                    c -> dies(c, lethal));
            final CardCollection theirsDead = CardLists.filter(
                    CardLists.getValidCards(ai.getOpponents().getCardsIn(ZoneType.Battlefield), valid, source.getController(), source, sa),
                    c -> dies(c, lethal));
            if (theirsDead.isEmpty()) {
                continue;
            }
            final int margin = ComputerUtilCard.evaluateCreatureList(theirsDead)
                    - ComputerUtilCard.evaluateCreatureList(oursDead);
            if (margin > bestMargin + MARGIN) {
                bestMargin = margin;
                bestX = x;
            }
        }
        return bestX;
    }

    private static boolean dies(final Card c, final int amount) {
        if (c.getNetToughness() <= amount) {
            return true;
        }
        return !c.hasKeyword(Keyword.INDESTRUCTIBLE) && ComputerUtilCombat.getDamageToKill(c, false) <= amount;
    }

    @Override
    public AiAbilityDecision chkDrawback(Player aiPlayer, SpellAbility sa) {
        return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
    }

    @Override
    protected AiAbilityDecision doTriggerNoCost(Player ai, SpellAbility sa, boolean mandatory) {
        // it might help so take it
        if (!sa.usesTargeting() && !sa.isCurse() && sa.hasParam("ValidCards") && sa.getParam("ValidCards").contains("YouCtrl")) {
            return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
        }

        // important to call canPlay first so targets are added if needed
        AiAbilityDecision decision = canPlay(ai, sa);
        if (mandatory && !decision.decision().willingToPlay()) {
            return new AiAbilityDecision(50, AiPlayDecision.MandatoryPlay);
        }
        return decision;
    }

    boolean pumpAgainstRemoval(Player ai, SpellAbility sa, List<Card> comp) {
        final List<GameObject> objects = ComputerUtil.predictThreatenedObjects(sa.getActivatingPlayer(), sa, true);
        for (final Card c : comp) {
            if (objects.contains(c)) {
                return true;
            }
        }
        return false;
    }
}
