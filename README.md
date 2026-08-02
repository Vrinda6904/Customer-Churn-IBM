## Final Output

This project successfully built a machine learning pipeline to predict customer churn, achieving strong accuracy with both Logistic Regression (85.8%) and Random Forest (85.3%) models. While Logistic Regression edged out slightly on overall accuracy, Random Forest was used for deeper evaluation due to its ability to capture non-linear patterns and provide feature importance insights.

The model performs well at identifying customers who stay (92% recall for class 0), but has more room to improve at catching customers who actually churn (66% recall for class 1).

Key churn drivers identified:
- Contract type – Month-to-month customers churn significantly more than those on long-term contracts.
- Tenure – Newer customers are more likely to leave.
- Monthly charge – Higher bills are linked to higher churn risk.
- Support services – Customers without tech support or security add-ons churn more often.

## Business Recommendation: To reduce churn, the company should prioritize converting month-to-month customers to longer contracts, strengthen onboarding for new customers, and offer tech support/security bundles. Especially to high-paying customers on flexible plans.
