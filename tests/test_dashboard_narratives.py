"""Narratives dashboard rendering checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

import app


def test_render_narratives_hides_empty_negative_drivers():
    narratives = {
        "countries": [
            {
                "country": "Greece",
                "probability": 0.9,
                "narrative": "Strong model signal.",
                "positive_drivers": [{"feature": "x", "shap_value": 0.2}],
                "negative_drivers": [],
            }
        ]
    }
    predictions_df = pd.DataFrame([{"country": "Greece", "probability": 0.92, "rank": 1}])
    column = MagicMock()

    with patch.object(app.st, "selectbox", return_value="Greece"), \
        patch.object(app.st, "columns", return_value=[column, column]), \
        patch.object(app.st, "subheader") as subheader, \
        patch.object(app.st, "dataframe"), \
        patch.object(app.st, "metric"), \
        patch.object(app.st, "write"), \
        patch.object(app.st, "markdown"), \
        patch.object(app, "render_page_header"), \
        patch.object(app, "_info_expander"):
        render_ctx = column.__enter__.return_value
        app.render_narratives(narratives, predictions_df)

    subheader.assert_called_once_with("Positive drivers")
