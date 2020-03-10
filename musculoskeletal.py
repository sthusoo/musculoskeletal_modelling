
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp


class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    beta = 0.1 # damping coefficient (see damped model in Millard et al.)
    # WRITE CODE HERE TO CALCULATE VELOCITY

    initial_velocity = 0
    
    def set_velocity(vm):
        f_l = np.array(force_length_muscle(lm))
        f_pe = np.array(force_length_parallel(lm))
        f_t = np.array(force_length_tendon(lt))
        func = (a*f_l*force_velocity_muscle(vm) + f_pe + beta*vm) - f_t
        return func

    velocity = fsolve(set_velocity, initial_velocity)

    return velocity


def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """
    lts = 1 

    if type(lt) == np.float64:
        if lt >= lts:
            tension = 10*(lt - lts) + 240*((lt - lts)**2)
            return tension
        else: 
            tension = 0
            return tension
    else:
        tension = []
        for i in lt:
            if i >= lts:
                tension.append(10*(i - lts) + 240*((i - lts)**2))
            else: 
                tension.append(0)

        return tension


def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """
    lpes = 1

    if type(lm) == np.float64:
        if lm >= lpes:
            force = 3*((lm - lpes)**2) / (.6 + lm - lpes)
            return force
        else: 
            force = 0
            return force
    else:
        force = []
        for i in lm:
            if i >= lpes:
                force.append(3*((i - lpes)**2) / (.6 + i - lpes))
            else: 
                force.append(0)

    return force


def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.show()


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.

    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """
    data = np.array([
        [37.42345184227537, 9.877204228297629],
        [38.43235940530059, 14.3481694015327],
        [39.28605042016807, 24.24870949465756],
        [39.44126696832579, 3.417558158104839],
        [40.42430510665805, 36.439472983763636],
        [40.415681965093725, 21.758760459840033],
        [40.415681965093725, 17.681367149574754],
        [41.38147382029735, 31.6210643750713],
        [41.41596638655462, 14.720247968870794],
        [41.459082094376214, 26.588801855583824],
        [41.82125404007757, 1.6124302825202506],
        [41.990842490842496, 31.79999738055099],
        [42.51972850678733, 41.97853166334406],
        [42.31277310924369, 31.51778957374806],
        [42.812915319974145, 46.397630645621746],
        [42.94398707175178, 23.620442617590015],
        [42.916393018745964, 48.59122943838169],
        [43.21820297349709, 50.349969884786475],
        [43.42515837104072, 44.50180600022816],
        [43.42280660515955, 34.633662634942894],
        [43.45102779573368, 53.75092000963281],
        [43.502766645119586, 23.442120661345598],
        [43.72121956474898, 56.681569951660904],
        [43.98135100193924, 59.078263051827065],
        [43.96841628959276, 22.021551302874073],
        [44.408196509372985, 45.46658970873416],
        [44.434065934065934, 60.443058375688125],
        [45.25790761274924, 53.69787533307789],
        [45.57601625265491, 45.58987307114184],
        [45.572320620555914, 43.04121490043981],
        [45.72753716871364, 67.31091925375661],
        [45.62405946994183, 53.07473044602456],
        [45.88275371687136, 70.71108391954067],
        [46.45732725478855, 72.21247965401928],
        [46.4260116354234, 62.67959131525913],
        [46.69764059469942, 75.49051085592608],
        [46.7105753070459, 44.51051214529414],
        [47.141732385261804, 80.55075792549297],
        [47.409049773755655, 71.38386649682496],
        [47.409049773755655, 62.56911365451157],
        [47.67291790562379, 81.60501822122515],
        [47.71085972850679, 66.82099877054894],
        [48.211001939237235, 83.29613632964498],
        [48.91809954751131, 85.0424154910833],
        [48.991396250808016, 81.16748001825164],
        [48.98708468002586, 62.62668119423829],
        [49.47860374919198, 76.02983763641203],
        [49.648602825745684, 85.82465037236614],
        [49.724363283775055, 81.93861713373131],
        [50.2598603749192, 84.27654356439407],
        [50.28055591467356, 87.47918123991678],
        [50.02186166774402, 81.21305585763716],
        [50.57374272786038, 74.44387845756704],
        [50.668597285067875, 90.3086948553177],
        [50.7092492381568, 77.84194373496004],
        [50.77207498383969, 84.94076749251735],
        [50.756553329023916, 86.94463486824594],
        [51.08051812440952, 79.03619254315021],
        [51.28084033613446, 77.73630662762842],
        [51.52626821142658, 89.48295686586503],
        [51.806851971557855, 91.02890471120574],
        [51.832721396250804, 88.7091020951367],
        [52.634673561732384, 89.09070147153884],
        [53.2641628959276, 88.65929077151222],
        [53.48466894450088, 93.91633545506113],
        [53.54010342598578, 83.2378371294219],
        [53.553038138332255, 78.85943277310923],
        [53.71749376673746, 92.06408387065233],
        [53.758998558003086, 96.52592641177597],
        [54.204085326438275, 93.93146905700439],
        [54.10923076923076, 91.25404479257766],
        [54.65248868778281, 99.27275485759915],
        [54.52314156431804, 93.82090041446443],
        [55.76487394957984, 95.86547033589248],
        [56.204654169360055, 95.91674651761156],
        [56.437478991596635, 99.49394074437672],
        [56.97392916680842, 98.92822965051703],
        [57.55503813833225, 98.9755941529553],
        [57.8085585003232, 90.862680976463],
        [57.963775048480926, 99.08213270466558],
        [58.44379659556131, 90.67185310806073],
        [58.55877181641887, 95.95726924434932],
        [58.92094376212023, 98.58080538838323],
        [59.36072398190045, 90.8114493706985],
        [59.49524499030382, 97.59031906916613],
        [59.68963523871086, 95.29820760702471],
        [60.05919844861022, 95.24341442640404],
        [60.550717517776334, 99.27589976805201],
        [60.58952165481577, 93.4280655918476],
        [61.300930833872016, 84.29951340009471],
        [61.34577117000646, 94.00659892264596],
        [61.352669683257915, 91.44578288147838],
        [61.40440853264383, 86.81936598349746],
        [61.39751001939237, 78.75038469396809],
        [61.4191910610398, 76.26522766210556],
        [61.40440853264383, 87.88432396669074],
        [62.18049127343245, 89.26176707721345],
        [62.30983839689722, 96.13781174189133],
        [62.56853264382676, 79.43536986197194],
        [63.1635294117647, 85.70883446657149],
        [63.34461538461538, 58.98564036655385],
        [63.353238526179695, 52.6903074134631],
        [63.42222365869425, 79.56858177422552],
        [63.67621437386143, 89.26395283609125],
        [63.72295572074984, 85.52542011374685],
        [63.93961215255332, 75.69785129749143],
        [63.951109674639085, 79.9373325136993],
        [64.48287007110537, 86.5061937716263],
        [64.50873949579832, 81.50091661279896],
        [64.72980548862901, 52.37577640352728],
        [65.10373626373627, 75.81798733792158],
        [65.3624305106658, 63.91082375755731],
        [65.5952553329024, 72.42445142400851],
        [65.70196670976082, 66.47237234305487],
        [65.67286360698125, 47.441536408228444],
        [66.13851325145443, 71.84436761854062],
        [66.00916612798966, 65.59461895889577],
        [66.31959922430511, 75.21845073957184],
        [66.86285714285714, 66.41296777444009],
        [67.14742081447963, 63.33921092056731],
        [67.09568196509373, 42.66710582151411],
        [67.38024563671623, 35.340935472831646],
        [67.48372333548804, 51.2745228335678],
        [67.8458952811894, 62.373190805734055],
        [68.4615875888817, 59.423300307996485],
        [68.5185003232062, 26.8260143351458],
        [69.37219133807369, 41.6392089018248],
        [70.20001292824821, 29.44353470158117],
        [70.53631544925662, 48.09978645575876],
        [70.56218487394958, 28.866494797013843],
        [71.39000646412411, 34.292947094428044],
        [72.41960956690369, 24.35181690558575],
        [72.9680413703943, 25.235577322331622],
        [73.40782159017454, 34.28119748763346],
        [73.40782159017454, 17.689074178416718],
        [73.40782159017454, 12.058854432764463],
        [74.39085972850678, 12.345192432327352],
        [75.19281189398836, 17.24694680406097],
        [75.399767291532, 11.987312331267333],
        [76.38280542986425, 7.95660234229436],
    ])
    
    max_tension = max(data[:,1])
    opt_length = float()

    for i, point in enumerate(data):
        if point[1] == max_tension:
            opt_length = point[0]

    norm_length = data[:,0] / opt_length
    norm_tension = data[:,1] / max_tension

    centres = np.arange(min(norm_length), max(norm_length), .2)
    width = .15
    result = Regression(norm_length, norm_tension, centres, width, .1, sigmoids=False)
    
    return result

force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))

# Question 1
plot_curves()

# Question 2
print("Velocity for a=1, lm=1, lt=1.01: {}".format(get_velocity([1], [1], [1.01])))

# Question 3
muscle = HillTypeMuscle(100, 0.3, 0.1) 

def f(t, x):
    lm = x
    lt = muscle.norm_tendon_length(0.4, lm)
    if t < 0.5:
        a = 0
    else:
        a = 1
        
    return get_velocity(a, lm, lt)

sol = solve_ivp(f, [0, 2], [1], max_step=.01)

forces = []
for i in sol.y[0]:
    forces.append(muscle.get_force(0.4, i))

plt.figure()
plt.subplot(1,2,1)
plt.plot(sol.t, sol.y.T)
plt.xlabel('Time (s)')
plt.ylabel('Normalized CE length')
plt.subplot(1,2,2)
plt.plot(sol.t, forces)
plt.xlabel('Time (s)')
plt.ylabel('Normalized Tension')
plt.tight_layout()
plt.show()
