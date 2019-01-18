import pytest

from sklearn.utils.estimator_checks import check_estimator

from UnexpectedPatternCoocurrence import TemplateEstimator
from UnexpectedPatternCoocurrence import TemplateClassifier
from UnexpectedPatternCoocurrence import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
