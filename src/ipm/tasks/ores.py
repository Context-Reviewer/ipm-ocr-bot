from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from ..decisions import choose_ore_sale
from .base import TaskResult


@dataclass(slots=True)
class OresTask:
    ore_reader: object | None = None
    sell_reader: object | None = None
    state_reader: object | None = None
    actions: object | None = None
    config: object | None = None
    name: str = "ores"

    @staticmethod
    def _verified(before_snapshot, after_snapshot, row_index: int) -> bool:
        if before_snapshot is None or after_snapshot is None:
            return False
        before_row = before_snapshot.ore_rows.get(int(row_index))
        after_row = after_snapshot.ore_rows.get(int(row_index))
        if before_row is None or after_row is None:
            return False
        if before_row.quantity is None or after_row.quantity is None:
            return False
        return int(after_row.quantity) < int(before_row.quantity)

    def _collect_snapshots(self, count: int):
        snapshots = []
        for _ in range(max(1, count)):
            snapshots.append(self.state_reader.read())
        return snapshots

    def _confirm_decision(self, decision, snapshots):
        valid_rows = []
        for snapshot in snapshots:
            row = snapshot.ore_rows.get(int(decision.row_index))
            if row is None or row.quantity is None or not row.ore_name:
                continue
            if row.ore_name != decision.ore_name:
                continue
            valid_rows.append(row)
        if not valid_rows:
            return None
        quantities = sorted(int(row.quantity) for row in valid_rows if row.quantity is not None)
        if not quantities:
            return None
        smallest = max(1, quantities[0])
        largest = quantities[-1]
        max_spread = float(getattr(self.config.policy, "ore_sell_max_relative_spread", 0.35))
        if ((largest - smallest) / smallest) > max_spread:
            return None
        median_qty = quantities[len(quantities) // 2]
        keep_qty = int(self.config.policy.ore_keep_quantity)
        sellable = max(0, median_qty - keep_qty)
        fraction = max(0.25, min(1.0, sellable / max(1, median_qty)))
        return replace(decision, quantity=median_qty, fraction=fraction)

    def run(self) -> TaskResult:
        if self.ore_reader is None or self.state_reader is None or self.actions is None or self.config is None:
            return TaskResult(
                ok=True,
                details={
                    "implemented": False,
                    "message": "Ore task dependencies not configured.",
                },
            )
        self.actions.reset_ui()
        self.actions.open_ores_panel()
        snapshot = self.state_reader.read()
        decision = choose_ore_sale(snapshot, self.config)
        executed = False
        verified = False
        after_snapshot = None
        action_results: dict[str, bool] = {}
        if decision is not None:
            confirm_reads = max(1, int(getattr(self.config.policy, "ore_sell_confirm_reads", 1)))
            confirmations = [snapshot]
            if confirm_reads > 1:
                confirmations.extend(self._collect_snapshots(confirm_reads - 1))
            confirmed = self._confirm_decision(decision, confirmations)
            action_results["confirmed"] = confirmed is not None
            if confirmed is None:
                decision = None
            else:
                decision = confirmed
                snapshot = confirmations[-1]
                action_results["select_row"] = bool(self.actions.select_ore_row(decision.row_index))
                action_results["open_sell_dialog"] = bool(
                    action_results["select_row"] and self.actions.open_sell_dialog()
                )
                action_results["choose_fraction"] = bool(
                    action_results["open_sell_dialog"] and self.actions.choose_sell_fraction(decision.fraction)
                )
                action_results["execute_sell"] = bool(
                    action_results["choose_fraction"] and self.actions.execute_sell()
                )
                executed = (
                    action_results.get("confirmed", False)
                    and action_results["select_row"]
                    and action_results["open_sell_dialog"]
                    and action_results["choose_fraction"]
                    and action_results["execute_sell"]
                )
                if executed:
                    after_reads = self._collect_snapshots(confirm_reads)
                    after_snapshot = after_reads[-1]
                    verified = any(self._verified(snapshot, candidate, decision.row_index) for candidate in after_reads)
        if self.sell_reader is not None:
            snapshot.sell_dialog = self.sell_reader.read()
            if after_snapshot is not None:
                after_snapshot.sell_dialog = self.sell_reader.read()
        self.actions.close_ores_panel()
        return TaskResult(
            ok=(decision is None) or (executed and verified),
            details={
                "implemented": True,
                "ore_rows": {row: asdict(state) for row, state in snapshot.ore_rows.items()},
                "sell_dialog": asdict(snapshot.sell_dialog),
                "decision": asdict(decision) if decision is not None else None,
                "action_results": action_results,
                "executed": executed,
                "verified": verified,
                "after_ore_rows": (
                    {row: asdict(state) for row, state in after_snapshot.ore_rows.items()}
                    if after_snapshot is not None
                    else None
                ),
            },
        )
