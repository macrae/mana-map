package forge.ai.ability;

import forge.ai.*;
import forge.card.mana.ManaCost;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;

public class ChangeTargetsAi extends SpellAbilityAi {

    /*
     * (non-Javadoc)
     * 
     * @see forge.ai.SpellAbilityAi#checkApiLogic(forge.game.player.Player,
     * forge.game.spellability.SpellAbility)
     */
    @Override
    protected AiAbilityDecision checkApiLogic(Player ai, SpellAbility sa) {
        final Game game = sa.getHostCard().getGame();
        final SpellAbility topSa = game.getStack().isEmpty() ? null
                : ComputerUtilAbility.getTopSpellAbilityOnStack(game, sa);

        if ("Self".equals(sa.getParam("DefinedMagnet"))) {
            return doSpellMagnet(sa, topSa, ai);
        }
        if ("Deflect".equals(sa.getParam("AILogic"))) {
            return doDeflectLogic(sa, topSa, ai);
        }

        // The AI can't otherwise play this ability, but should at least not
        // miss mandatory activations (e.g. triggers).
        if (sa.isMandatory()) {
            return new AiAbilityDecision(50, AiPlayDecision.MandatoryPlay);
        }
        return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
    }

    // MANA MAP PATCH (2026-09-30): AILogic$ Deflect — Deflecting Swat / Bolt Bend. The shipped AI
    // "can't otherwise play this ability" and never cast Swat in 80 games while 17 opposing spells
    // or abilities were aimed at our seat or our permanents. Fire when the top of the stack is an
    // opponent's targeted spell or ability aimed at us or at something we control, and there is
    // somewhere else legal to send it; the new target is chosen by PlayerControllerAi.chooseNewTargetsFor
    // (patched alongside — it used to return null).
    private AiAbilityDecision doDeflectLogic(SpellAbility sa, SpellAbility topSa, Player ai) {
        if (topSa == null || !topSa.usesTargeting()) {
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        final Card topHost = topSa.getHostCard();
        if (topHost == null || !topHost.getController().isOpponentOf(ai)) {
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        final TargetChoices tgts = topSa.getTargets();
        boolean atUs = tgts.contains(ai);
        for (Card c : tgts.getTargetCards()) {
            if (c.getController().equals(ai)) {
                atUs = true;
            }
        }
        if (!atUs || !sa.canTarget(topSa)) {
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        boolean elsewhere = false;
        if (topSa.getTargetRestrictions() != null) {
            for (GameEntity e : topSa.getTargetRestrictions().getAllCandidates(topSa)) {
                if (e instanceof Card && ((Card) e).getController().equals(ai)) {
                    continue;
                }
                if (e instanceof Player && e.equals(ai)) {
                    continue;
                }
                if (topSa.canTarget(e)) {
                    elsewhere = true;
                    break;
                }
            }
        }
        if (!elsewhere) {
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        sa.resetTargets();
        sa.getTargets().add(topSa);
        return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
    }

    private AiAbilityDecision doSpellMagnet(SpellAbility sa, SpellAbility topSa, Player aiPlayer) {
        // For cards like Spellskite that retarget spells to itself
        if (topSa == null) {
            // nothing on stack, so nothing to target
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        final TargetChoices topTargets = topSa.getTargets();
        final Card topHost = topSa.getHostCard();

        if (!sa.getTargets().isEmpty() && sa.isTrigger()) {
            // something was already chosen before (e.g. in response to a trigger - Mizzium Meddler), so just proceed
            return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
        }

        if (!topSa.usesTargeting() || topTargets.getTargetCards().contains(sa.getHostCard())) {
            // if this does not target at all or already targets host, no need to redirect it again
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }

        for (Card tgt : topTargets.getTargetCards()) {
            if (ComputerUtilAbility.getAbilitySourceName(sa).equals(tgt.getName()) && tgt.getController().equals(aiPlayer)) {
                // We are already targeting at least one card with the same name (e.g. in presence of 2+ Spellskites),
                // no need to retarget again to another one
                return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
            }
        }

        if (topHost != null && !topHost.getController().isOpponentOf(aiPlayer)) {
            // make sure not to redirect our own abilities
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        if (!topSa.canTarget(sa.getHostCard())) {
            // don't try targeting it if we can't legally target the host card with it in the first place
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        if (!sa.canTarget(topSa)) {
            // don't try retargeting a spell that the current card can't legally retarget (e.g. Muck Drubb + Lightning Bolt to the face)
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }

        if (sa.getPayCosts().getCostMana() != null && sa.getPayCosts().getCostMana().getMana().hasPhyrexian()) {
            ManaCost manaCost = sa.getPayCosts().getCostMana().getMana();
            int payDamage = manaCost.getPhyrexianCount() * 2;
            // e.g. Spellskite or a creature receiving its ability that requires Phyrexian mana P/U
            int potentialDmg = ComputerUtil.predictDamageFromSpell(topSa, aiPlayer);
            ManaCost normalizedMana = manaCost.getNormalizedMana();
            boolean canPay = ComputerUtilMana.canPayManaCost(new ManaCostBeingPaid(normalizedMana), sa, aiPlayer, false);
            if (potentialDmg != -1 && potentialDmg <= payDamage && !canPay
                    && topTargets.contains(aiPlayer)) {
                // do not pay Phyrexian mana if the spell is a damaging one but it deals less damage or the same damage as we'll pay life
                return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
            }
        }

        Card firstCard = topTargets.getFirstTargetedCard();
        // if we're not the target don't intervene unless we can steal a buff
        if (firstCard != null && !aiPlayer.equals(firstCard.getController()) && !topHost.getController().equals(firstCard.getController()) && !topHost.getController().getAllies().contains(firstCard.getController())) {
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }
        Player firstPlayer = topTargets.getFirstTargetedPlayer();
        if (firstPlayer != null && !aiPlayer.equals(firstPlayer)) {
            return new AiAbilityDecision(0, AiPlayDecision.CantPlayAi);
        }

        sa.resetTargets();
        sa.getTargets().add(topSa);
        return new AiAbilityDecision(100, AiPlayDecision.WillPlay);
    }
}
