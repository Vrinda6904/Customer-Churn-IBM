## Final Output

This project developed an end-to-end machine learning pipeline to predict customer churn using the IBM Telco Customer Churn dataset. Logistic Regression achieved 85.8% accuracy, while Random Forest achieved 85.3%. Although Logistic Regression performed slightly better in overall accuracy, Random Forest was explored further for its ability to capture non-linear relationships and identify important features influencing churn.

The evaluation showed that the models were more effective at identifying customers who stayed (92% recall for class 0) than customers who churned (66% recall for class 1), highlighting opportunities to improve churn detection.

**Key factors influencing customer churn:**
- **Contract Type:** Customers on month-to-month contracts were more likely to churn than those with long-term contracts.
- **Tenure:** Customers with shorter relationships with the company showed a higher likelihood of leaving.
- **Monthly Charges:** Higher monthly bills were associated with increased churn risk.
- **Support Services:** Customers without technical support or online security services were more likely to churn.

**Business Recommendations:** The company should encourage month-to-month customers to switch to longer-term contracts, strengthen onboarding and engagement for new customers, and offer technical support and online security bundles. Targeted retention offers for high-paying customers on flexible contracts could also help reduce churn.
