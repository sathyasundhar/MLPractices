from pgmpy.models import BayesianModel

AM = BayesianModel([('Burglary', 'Alarm'), 
                              ('Earthquake', 'Alarm'),
                              ('Alarm', 'JohnCalls'),
                              ('Alarm', 'MaryCalls')])

from pgmpy.factors.discrete import TabularCPD

cb = TabularCPD(variable='Burglary', variable_card=2,
                      values=[[.999], [0.001]])
ce = TabularCPD(variable='Earthquake', variable_card=2,
                       values=[[0.998], [0.002]])
ca = TabularCPD(variable='Alarm', variable_card=2,
                        values=[[0.999, 0.71, 0.06, 0.05],
                                [0.001, 0.29, 0.94, 0.95]],
                        evidence=['Burglary', 'Earthquake'],
                        evidence_card=[2, 2])
cj = TabularCPD(variable='JohnCalls', variable_card=2,
                      values=[[0.95, 0.1], [0.05, 0.9]],
                      evidence=['Alarm'], evidence_card=[2])
cm = TabularCPD(variable='MaryCalls', variable_card=2,
                      values=[[0.1, 0.7], [0.9, 0.3]],
                      evidence=['Alarm'], evidence_card=[2])

AM.add_cpds(cb, ce, ca, cj, cm)


AM.check_model()


AM.nodes()


AM.edges()


AM.get_independencies()


cpd_j_values=cj.get_values()
cpd_m_values=cm.get_values()
cpd_a_values=ca.get_values()
cpd_b_values=cb.get_values()
cpd_e_values=ce.get_values()
cpd_j_values[1][1]*cpd_m_values[0][1]*cpd_a_values[1][0]*cpd_b_values[0][0]*cpd_e_values[0][0]


