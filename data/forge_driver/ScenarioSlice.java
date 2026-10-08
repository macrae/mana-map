// THE SCENARIO-SLICE RUNNER (PRD v2 Step 5, 2026-10-08): one board, a few turns, AI on
// every seat, many seeds in ONE JVM. `manamap.sim.slice` builds it into its own small jar
// (mm-scenario-slice.jar) that rides on the classpath BESIDE the Forge jar a run already
// uses, so the telemetry jar, its `-tl<sha8>` fingerprint and every pod run stay exactly
// as they were: this class never executes in a `sim` game.
//
//   java -cp <forge jar>:mm-scenario-slice.jar mm.ScenarioSlice \
//        --decks a.dck,b.dck[,…] --case A:1=a1.state --case A:2=a2.state [--case B:1=…] \
//        [--rounds 1] [--timeout 120]
//
// A state file is Forge's puzzle `[state]` block (p0..pN keys; GameState.parseLine).
// ONE BOARD PER (ARM, SEED): the converter draws each seat's hidden hand and library
// from its decklist per seed, so a replicate is a different deal, not the same board
// replayed — with the deal fixed, Forge played three seeds of a lifted board as the
// identical game (2026-10-08).
// For each case: a fresh Commander match of AI players from the decks,
// the state applied at the first priority, played until the turn AFTER the current turn
// plus `rounds` full rounds begins, then snapshotted. One line per replicate on stdout:
// `MMSLICE {json}`. Everything else Forge prints is noise to the reader.
//
// THE TRAP THE SPIKE FOUND. GameState.applyToGame() hands its work to Forge's game-thread
// executor when the caller is not one of its threads — and a sim's thread is not — so the
// state raced the game and half-applied (zones cleared, never filled). The start hook runs
// on the thread that then runs the main loop, so the protected apply is called directly.
package mm;

import com.google.common.eventbus.Subscribe;
import forge.GuiDesktop;
import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.Game;
import forge.game.GameEndReason;
import forge.game.GameLogEntry;
import forge.game.GameRules;
import forge.game.GameState;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.event.GameEventTurnBegan;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.zone.ZoneType;
import forge.gui.GuiBase;
import forge.localinstance.properties.ForgeConstants;
import forge.model.FModel;
import forge.player.GamePlayerUtil;
import forge.util.MyRandom;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class ScenarioSlice {

    static class SliceState extends GameState {
        void applyNow(Game g) { applyGameOnThread(g); }
    }

    /** Stops the game when the turn after the slice's last turn begins. */
    static class Stopper {
        final Game game;
        final int seats, rounds;
        int startTurn = -1, stopTurn = -1, endedAtTurn = -1;
        boolean stopped = false;
        Snapshot snap;

        Stopper(Game game, int seats, int rounds) {
            this.game = game; this.seats = seats; this.rounds = rounds;
        }

        void start(int turn) {
            startTurn = turn;
            stopTurn = turn + seats * rounds;         // the last turn the slice plays
        }

        @Subscribe
        public void onTurn(GameEventTurnBegan ev) {
            if (startTurn < 0 || stopped || ev.turnNumber() <= stopTurn) return;
            stopped = true;
            endedAtTurn = ev.turnNumber() - 1;
            snap = Snapshot.of(game);                 // the board as the last turn left it
            game.setGameOver(GameEndReason.Draw);
        }
    }

    static class Snapshot {
        final List<Map<String, Object>> seats = new ArrayList<>();

        static Snapshot of(Game g) {
            Snapshot s = new Snapshot();
            int i = 0;
            // REGISTERED players: getPlayers() drops a seat once it has lost, and a
            // four-seat snapshot came back with three.
            for (Player p : g.getRegisteredPlayers()) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("seat", "p" + i++);
                m.put("name", p.getName());
                m.put("life", p.getLife());
                m.put("lost", p.hasLost());
                m.put("hand", names(p, ZoneType.Hand));
                m.put("battlefield", names(p, ZoneType.Battlefield));
                m.put("graveyard", names(p, ZoneType.Graveyard));
                m.put("exile", names(p, ZoneType.Exile));
                m.put("command", names(p, ZoneType.Command));
                m.put("library_count", p.getCardsIn(ZoneType.Library).size());
                s.seats.add(m);
            }
            return s;
        }

        static List<String> names(Player p, ZoneType z) {
            List<String> out = new ArrayList<>();
            for (Card c : p.getCardsIn(z)) out.add(c.getName());
            return out;
        }
    }

    public static void main(String[] argv) throws Exception {
        Map<String, List<String>> a = parse(argv);
        List<String> decks = split(one(a, "decks"));
        int rounds = Integer.parseInt(a.containsKey("rounds") ? one(a, "rounds") : "1");
        int timeout = Integer.parseInt(a.containsKey("timeout") ? one(a, "timeout") : "120");

        GuiBase.setInterface(new GuiDesktop());
        FModel.initialize(null, null);

        List<Deck> loaded = new ArrayList<>();
        for (String d : decks) {
            File f = new File(ForgeConstants.DECK_COMMANDER_DIR + d);
            if (!f.exists()) { fail("no deck " + f); return; }
            loaded.add(DeckSerializer.fromFile(f));
        }
        ExecutorService pool = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "slice"); t.setDaemon(true); return t;
        });
        for (String spec : a.get("case")) {
            int eq = spec.indexOf('='), colon = spec.lastIndexOf(':', eq);
            if (eq < 0 || colon < 0) { fail("a --case is LABEL:SEED=path, got " + spec); return; }
            String label = spec.substring(0, colon);
            long seed = Long.parseLong(spec.substring(colon + 1, eq));
            List<String> lines = Files.readAllLines(Paths.get(spec.substring(eq + 1)));
            {
                Future<String> f = pool.submit(() -> replicate(label, lines, seed, loaded, rounds));
                String out;
                try {
                    out = f.get(timeout, TimeUnit.SECONDS);
                } catch (TimeoutException e) {
                    f.cancel(true);
                    out = "{\"label\":" + q(label) + ",\"seed\":" + seed + ",\"error\":\"timeout\"}";
                } catch (Exception e) {
                    out = "{\"label\":" + q(label) + ",\"seed\":" + seed + ",\"error\":" + q(String.valueOf(e.getCause())) + "}";
                }
                System.out.println("MMSLICE " + out);
                System.out.flush();
            }
        }
        System.exit(0);
    }

    static String replicate(String label, List<String> lines, long seed, List<Deck> decks, int rounds) {
        MyRandom.setRandom(new Random(seed));
        GameRules rules = new GameRules(GameType.Commander);
        rules.setAppliedVariants(EnumSet.of(GameType.Commander));
        List<RegisteredPlayer> pp = new ArrayList<>();
        for (int i = 0; i < decks.size(); i++) {
            RegisteredPlayer rp = RegisteredPlayer.forCommander(decks.get(i));
            rp.setPlayer(GamePlayerUtil.createAiPlayer("Ai(" + (i + 1) + ")-" + decks.get(i).getName(), i, ""));
            pp.add(rp);
        }
        Match mc = new Match(rules, pp, "Slice");
        Game g = mc.createGame();
        SliceState state = new SliceState();
        state.parse(lines);
        Stopper stop = new Stopper(g, decks.size(), rounds);
        g.subscribeToEvents(stop);
        Snapshot[] start = new Snapshot[1];
        long t0 = System.currentTimeMillis();
        mc.startGame(g, () -> {
            state.applyNow(g);
            stop.start(g.getPhaseHandler().getTurn());
            start[0] = Snapshot.of(g);
        });
        long ms = System.currentTimeMillis() - t0;
        Snapshot end = stop.snap != null ? stop.snap : Snapshot.of(g);
        String winner = null;
        if (!stop.stopped && g.getOutcome() != null && !g.getOutcome().isDraw()) {
            winner = g.getOutcome().getWinningLobbyPlayer().getName();
        }
        List<GameLogEntry> log = new ArrayList<>(g.getGameLog().getLogEntries(null));
        Collections.reverse(log);                     // Forge stores newest first
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"label\":").append(q(label)).append(",\"seed\":").append(seed)
          .append(",\"rounds\":").append(rounds)
          .append(",\"start_turn\":").append(stop.startTurn)
          .append(",\"stop_turn\":").append(stop.stopTurn)
          .append(",\"ended_at_turn\":").append(stop.stopped ? stop.endedAtTurn : g.getPhaseHandler().getTurn())
          .append(",\"stopped\":").append(stop.stopped)
          .append(",\"winner\":").append(winner == null ? "null" : q(winner))
          .append(",\"elapsed_ms\":").append(ms)
          .append(",\"start\":").append(json(start[0] == null ? null : start[0].seats))
          .append(",\"end\":").append(json(end.seats))
          .append(",\"log\":[");
        for (int i = 0; i < log.size(); i++) {
            if (i > 0) sb.append(',');
            GameLogEntry e = log.get(i);
            sb.append(q(e.toString()));             // the line `sim` prints and parse.py reads
        }
        return sb.append("]}").toString();
    }

    // ── tiny helpers: argv, JSON ────────────────────────────────────────────

    static Map<String, List<String>> parse(String[] argv) {
        Map<String, List<String>> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < argv.length; i += 2) {
            m.computeIfAbsent(argv[i].replaceFirst("^--", ""), k -> new ArrayList<>()).add(argv[i + 1]);
        }
        for (String need : new String[]{"decks", "case"}) {
            if (!m.containsKey(need)) { fail("missing --" + need); }
        }
        return m;
    }

    static String one(Map<String, List<String>> m, String k) { return m.get(k).get(0); }

    static List<String> split(String s) {
        List<String> out = new ArrayList<>();
        for (String x : s.split(",")) if (!x.trim().isEmpty()) out.add(x.trim());
        return out;
    }

    static void fail(String why) {
        System.out.println("MMSLICE {\"error\":" + q(why) + "}");
        System.exit(2);
    }

    static String q(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c)); else b.append(c);
            }
        }
        return b.append('"').toString();
    }

    @SuppressWarnings("unchecked")
    static String json(Object o) {
        if (o == null) return "null";
        if (o instanceof String) return q((String) o);
        if (o instanceof Number || o instanceof Boolean) return o.toString();
        if (o instanceof Map) {
            StringBuilder b = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<String, Object> e : ((Map<String, Object>) o).entrySet()) {
                if (!first) b.append(',');
                first = false;
                b.append(q(e.getKey())).append(':').append(json(e.getValue()));
            }
            return b.append('}').toString();
        }
        if (o instanceof List) {
            StringBuilder b = new StringBuilder("[");
            List<Object> l = (List<Object>) o;
            for (int i = 0; i < l.size(); i++) { if (i > 0) b.append(','); b.append(json(l.get(i))); }
            return b.append(']').toString();
        }
        return q(String.valueOf(o));
    }
}
