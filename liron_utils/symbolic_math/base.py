import typing

import sympy


class SymExpr:
    """Symbolic expression with optional substitution values, uncertainties, and units."""

    def __init__(
        self,
        expression: sympy.Expr | str,
        *,
        subs_value: dict[str, typing.Any] | None = None,
        dev: dict[str, typing.Any] | None = None,
        units: str | None = None,
        info: str | None = None,
    ) -> None:
        """Initialize the expression and its annotations.

        Args:
            expression: Sympy expression or string representation.
            subs_value: Map of symbol name to numeric value, e.g. ``{"x": 5, "y": 6}``.
            dev: Map of symbol name to standard deviation, e.g. ``{"x": 0.5}``.
            units: Optional units label, appended in ``subs(out_type=str)``.
            info: Free-form description of the quantity.
        """
        self.expr = expression

        self.subs_value = subs_value
        self.dev = dev
        self.units = units
        self.info = info

    def __call__(self) -> sympy.Expr | str:
        return self.expr

    def subs(self, out_type: type = sympy.Expr) -> tuple[typing.Any, typing.Any] | str:
        """Substitute ``self.subs_value`` into the expression and return the result.

        Args:
            out_type: Either ``sympy.Expr`` (returns ``(value, dev)``) or ``str``
                (returns ``"value +- dev [units]"``).

        Returns:
            Tuple ``(value, dev)`` when ``out_type is sympy.Expr``, else a string.

        Raises:
            AssertionError: If ``out_type`` is not ``sympy.Expr`` or ``str``.
        """
        assert out_type in (sympy.Expr, str)
        value: typing.Any = self.expr
        dev: typing.Any = self.dev

        if isinstance(self.expr, sympy.Expr):
            if self.subs_value is None:
                print("No values to substitute.")
            else:
                value = self.expr.subs(self.subs_value)
                dev = self.calc_dev()

        if out_type is sympy.Expr:
            out: tuple[typing.Any, typing.Any] | str = (value, dev)
        else:
            out = str(value)

            if dev is not None:
                out += f" +- {dev}"

            if self.units is not None:
                out += f" [{self.units}]"

        return out

    def calc_dev(self, subs_value: bool = True) -> sympy.Expr:
        """Compute the propagated standard deviation via the gradient method.

        Args:
            subs_value: If True, substitute ``self.subs_value`` into the result.

        Returns:
            The propagated deviation as a Sympy expression.

        Raises:
            AssertionError: If ``self.expr`` isn't a Sympy expression or ``self.dev`` is None.
        """
        assert isinstance(self.expr, sympy.Expr), "calc_dev requires a sympy expression"
        assert self.dev is not None, "calc_dev requires dev values"

        gradient: dict[typing.Any, typing.Any] = {sym: self.expr.diff(sym) for sym in self.expr.free_symbols}

        out: sympy.Expr = sympy.core.numbers.Zero()
        for sym in self.expr.free_symbols:
            if sym in self.dev and self.dev[sym] is not None:
                out += (gradient[sym] * self.dev[sym]) ** 2

        out = sympy.sqrt(out)

        if subs_value and self.subs_value is not None:
            out = out.subs(self.subs_value)

        return out

    def __str__(self) -> str:
        return f"{self.expr}, dev={self.dev}"

    def __repr__(self) -> str:
        return self.__str__()
