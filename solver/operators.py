import math
from sympy.parsing.latex import parse_latex
from sympy.printing import latex
from sympy import solve
from sympy.core.numbers import Float

def g_equal(n1):  # 0
    return n1


def g_double(n1):  # 1
    return n1*2


def g_half(n1):  # 2
    return n1/2


def g_add(n1, n2):  # 3
    return n1 + n2


def g_minus(n1, n2):  # 4
    return math.fabs(n1 - n2)


def g_sin(n1):  # 5
    if n1 % 15 == 0 and 0 <= n1 <= 180:
        return math.sin(n1/180*math.pi)
    return False


def g_cos(n1):  # 6
    if n1 % 15 == 0 and 0 <= n1 <= 180:
        return math.cos(n1/180*math.pi)
    return False


def g_tan(n1):  # 7
    if n1 % 15 == 0 and 5 <= n1 <= 85:
        return math.tan(n1/180*math.pi)
    return False


def g_asin(n1):  # 8
    if -1 < n1 < 1:
        n1 = math.asin(n1)
        n1 = math.degrees(n1)
        return n1
    return False


def g_acos(n1):  # 9
    if -1 < n1 < 1:
        n1 = math.acos(n1)
        n1 = math.degrees(n1)
        return n1
    return False


def gougu_add(n1, n2):  # 13
    return math.sqrt(n1*n1+n2*n2)


def gougu_minus(n1, n2):  # 14
    if n1 != n2:
        return math.sqrt(math.fabs(n1*n1-n2*n2))
    return False


def g_bili(n1, n2, n3):  # 16
    if n1 > 0 and n2 > 0 and n3 > 0:
        return n1/n2*n3
    else:
        return False


def g_mul(n1, n2):  # 17
    return n1*n2


def g_divide(n1, n2):  # 18
    if n1 > 0 and n2 > 0:
        return n1/n2
    return False


def cal_circle_area(n1):  # 19
    return n1*n1*math.pi


def cal_circle_perimeter(n1):  # 20
    return 2*math.pi*n1


def cal_cone(n1, n2):  # 21
    return n1*n2*math.pi



################# Program Executor ########################

spec_token_list = ['frac', 'pi', 'sqrt']
spec_letter_list = ['f', 'r', 'a', 'c', 'p', 'i', 's', 'q', 'r', 't']
low_case_list = [chr(i) for i in range(97, 123)]
fixed_order_ops = [
    'Get', 'Iso_Tri_Ang', 'Gsin', 'Gcos', 'Gtan', 'Geo_Mean', 'Ratio', 'TanSec_Ang', \
    'Chord2_Ang', 'Tria_BH_Area', 'Para_Area', 'Kite_Area', 'Circle_R_Circum', \
    'Circle_D_Circum', 'Circle_R_Area', 'Circle_D_Area', 'ArcSeg_Area', 'Ngon_Angsum', \
    'RNgon_B_Area', 'RNgon_L_Area', 'RNgon_H_Area']
alterable_order_ops = [
    'Sum', 'Multiple', 'Equal', 'Gougu', 'Cos_Law', 'Sin_Law', 'Median', 'Proportion', \
    'Tria_SAS_Area', 'PRK_Perim', 'Rect_Area', 'Rhom_Area', 'Trap_Area']
arith_op_list = fixed_order_ops + alterable_order_ops
priority_list = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", \
                "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", \
                "C0.5", "C2", "C3", "C4", "C5", "C6", "C8", "C60", "C90", "C180", "C360"]
V_NUM = 10

class Varible_Record(object):
    def __init__(self):
        self.varible_dict = dict()
        self.mid_varible_dict = dict()
        self.result = ''

def get_priority(token):
    if token in priority_list:
        return priority_list.index(token)
    else:
        return -1 # arg

def is_exist_operator(func, ANNO):
    if not func in arith_op_list:
        print("Can Not Find Operators!")
        raise Exception
    return func

def choose_result(result_list):
    if len(result_list)==0:
        return None
    elif len(result_list)==1:
        return result_list[0]
    elif len(result_list)>1:
        t1 = result_list[0].evalf()
        t2 = result_list[1].evalf()
        if (t1>t2 and t2<=0) or (t1<t2 and t1>0):
            return result_list[0]
        else:
            return result_list[1]
        
def operand_update(operands, ANNO):
    for id in range(len(operands)):
        # Substitute variable
        if operands[id] in ANNO.mid_varible_dict:
            operands[id] = "("+ANNO.mid_varible_dict[operands[id]]+")"
            # pi
            if "\\pi" in operands[id]:
                operands[id]=operands[id].replace('\\pi','(3.141593)')
            # mixed number (improper fraction)
            if "\\frac" in operands[id]:
                loc = operands[id].index("\\frac")
                if loc>0 and operands[id][loc-1].isdigit():
                    operands[id] = operands[id][:loc]+'+'+operands[id][loc:]
            continue
        # Substitute process(intermediate) variable
        if operands[id] in ANNO.mid_varible_dict:
            operands[id] = "("+ANNO.mid_varible_dict[operands[id]]+")"
            continue 
        # Substitute constant
        if operands[id][0] == 'C':
            operands[id] = operands[id][1:]

    return operands

def mid_var_solve(expr_step, ANNO, visit_list, midvar2letter):

    # replace process(intermediate) variable
    for key, value in midvar2letter.items():
        expr_step = expr_step.replace(key, value)
    # Convert the latex form expression to sympy solvable form
    expr_step = parse_latex(expr_step)
    # Solving argument
    for letter in visit_list:
        try:
            result = solve(expr_step, letter)
            result = choose_result(result)
        except:
            ANNO.mid_varible_dict[letter] = letter
            continue
        if not result is None:
            result = latex(result)
            is_update = True
            result_t = result[:]
            for item in spec_token_list:
                result_t = result_t.replace(item, '')
            # more than one unknown varibles 
            for letter_t in visit_list:
                if letter_t in result_t and letter<letter_t:
                    is_update = False
                    break
            # intermediate variables are existed 
            for key, value in midvar2letter.items():
                if value in result_t:
                    is_update = False
                    break
            if is_update:
                ANNO.mid_varible_dict[letter] = result
            else:
                ANNO.mid_varible_dict[letter] = letter

    # Solving process(intermediate) variable
    for key1, value1 in midvar2letter.items():
        if value1 in str(expr_step):
            result = solve(expr_step, value1)
            result = choose_result(result)
            if not result is None:
                # Convert the intermediate variable to latex form
                result = latex(result)
                # Convert lowercase letters to intermediate variables V_i
                is_update = True
                # more than one intermediate variables, only take the front intermediate variables
                for key2, value2 in midvar2letter.items():
                    if value2 in result and value1<value2:
                        is_update = False
                        break
                    result.replace(value2, key2)
                if is_update:
                    ANNO.mid_varible_dict[key1] = result
                else:
                    ANNO.mid_varible_dict[key1] = key1

def mid_var_update(ANNO, visit_list, midvar2letter, midletter2var, is_subs_visit=True):

    has_solved_list = []
    # Find solved process varibles and arguments
    for key, value in ANNO.mid_varible_dict.items():
        if value!='' and isinstance(parse_latex(value).evalf(), Float) or \
                (key in visit_list and is_subs_visit):
            if not key in midvar2letter:
                has_solved_list.append(key)
            else:
                has_solved_list.append(midvar2letter[key])

    for key, mid_var in ANNO.mid_varible_dict.items():
        if value!='' and not key in has_solved_list:
            # Process varibles V_i are replaced as lowercase letters
            for key1, value1 in midvar2letter.items():
                mid_var = mid_var.replace(key1, value1)
            # Special characters are replaced with '@' for marking
            mid_var_t = mid_var[:]
            for item in spec_token_list:
                mid_var_t = mid_var_t.replace(item, "@"*len(item))
            # Lowercase letters are replaced with solved values
            mid_var_new = ''
            for id in range(len(mid_var_t)):
                if mid_var_t[id]!="@" and mid_var_t[id] in has_solved_list:
                    if mid_var_t[id] in midletter2var:
                        mid_var_new += "("+ANNO.mid_varible_dict[midletter2var[mid_var_t[id]]]+')'
                    else:
                        mid_var_new += "("+ANNO.mid_varible_dict[mid_var_t[id]]+')'
                else:
                    mid_var_new += mid_var[id]
            # Lowercase letters are replaced with V_i
            for key2, value2 in midvar2letter.items():
                mid_var_new = mid_var_new.replace(value2, key2)
            ANNO.mid_varible_dict[key] = mid_var_new

def Get(ANNO, arg_list):
    """
        Get(a) -> get numerical value of a
    """
    if len(arg_list)!=1:
        print("<Gets> function has only 1 augment!")
        raise Exception

    if arg_list[0] in ANNO.mid_varible_dict:
        result = ANNO.mid_varible_dict[arg_list[0]]
    else:
        result_v = ANNO.varible_dict[arg_list[0]]
        result_t = result_v[:]
        for item in spec_token_list:
            result_t = result_t.replace(item, "@"*len(item))
        # Lowercase letters are replaced with solved values
        result = ''
        for id in range(len(result_t)):
            if result_t[id]!="@" and result_t[id] in ANNO.mid_varible_dict:
                result += "("+ANNO.mid_varible_dict[result_t[id]]+')'
            else:
                result += result_v[id]
    ANNO.result = format(float(parse_latex(result).evalf()),'0.3f')

def Sum(arg_list):
    """
        Sum(a, b, c, d) -> a+b+c=d
    """
    if len(arg_list)<3:
        print("<Sum> function has 3 augments at least!")
        raise Exception
    expr_step = arg_list[0]
    for item in arg_list[1:-1]:
        expr_step += "+" + item
    expr_step += "-" + arg_list[-1]
    return expr_step

def Multiple(arg_list):
    """
        Multiple(a, b, c, d, e) -> a*b*c*d=e
    """
    if len(arg_list)<3:
        print("<Product> function has 3 augments at least!")
        raise Exception
    expr_step = arg_list[0]
    for item in arg_list[1:-1]:
        expr_step += "*" + item
    expr_step += "-" + arg_list[-1]
    return expr_step

def Equal(arg_list):
    """
        Equal(a, b) -> a=b
    """
    if len(arg_list)!=2:
        print("<Equal> function has 2 augments!")
        raise Exception
    expr_step = arg_list[0] + "-" + arg_list[-1]
    return expr_step

def Iso_Tri_Ang(arg_list):
    """
        Iso_Tri_Ang(a, b) -> a+2*b=180
    """
    if len(arg_list)!=2:
        print("<Iso_Tri_Ang> function has 2 augments!")
        raise Exception
    expr_step = arg_list[0] + "+2*" + arg_list[-1]+"-180"
    return expr_step

def Gougu(arg_list):
    """
        Gougu(a, b, c) -> a^2+b^2=c^2
    """
    if len(arg_list)!=3:
        print("<Gougu> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'^{2}'+"+"+arg_list[1]+"^{2}"+'-'+arg_list[2]+"^{2}"
    return expr_step

def Gsin(arg_list):
    """
        Gsin(a, b, c) -> sin(c)=a/b
    """
    if len(arg_list)!=3:
        print("<Gsin> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'/'+arg_list[1]+'-'+'\\sin{'+arg_list[2]+'/180*3.141593}'
    return expr_step

def Gcos(arg_list):
    """
        Gcos(a, b, c) -> cos(c)=a/b 
    """
    if len(arg_list)!=3:
        print("<Gcos> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'/'+arg_list[1]+'-'+'\\cos{'+arg_list[2]+'/180*3.141593}'
    return expr_step

def Gtan(arg_list):
    """
        Gtan(a, b, c) -> tan(c)=a/b 
    """
    if len(arg_list)!=3:
        print("<Gtan> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'/'+arg_list[1]+'-'+'\\tan{'+arg_list[2]+'/180*3.141593}'
    return expr_step

def Cos_Law(arg_list):
    """
        Cos_Law(a, b, c, d) -> a^2=b^2+c^2-2*b*c
    """
    if len(arg_list)!=4:
        print("<Cos_Law> function has 4 augments!")
        raise Exception
    expr_step = arg_list[1]+'^{2}'+"+"+arg_list[2]+"^{2}"+'-'+arg_list[0]+"^{2}"+ \
                '-'+"2*"+arg_list[1]+'*'+arg_list[2]+'*'+'\\cos{'+arg_list[3]+'/180*3.141593}'
    return expr_step

def Sin_Law(arg_list):
    """
        Sin_Law(a, b, c, d) -> sin(a)/b=sin(c)/d
    """
    if len(arg_list)!=4:
        print("<Sin_Law> function has 4 augments!")
        raise Exception
    expr_step = arg_list[3]+'*'+'\\sin{'+arg_list[0]+'/180*3.141593}'+'-'+ \
                    arg_list[1]+'*'+'\\sin{'+arg_list[2]+'/180*3.141593}'
    return expr_step

def Median(arg_list):
    """
        Median(a, b, c) -> a+c=2*b        
    """
    if len(arg_list)!=3:
        print("<Median> function has 3 augments!")
        raise Exception 
    expr_step = arg_list[0]+'-2*'+arg_list[1]+"+"+arg_list[2]
    return expr_step

def Geo_Mean(arg_list):
    """
        Geo_Mean(a, b, c) -> a*b=c^2
    """
    if len(arg_list)!=3:
        print("<Geo_Mean> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[1]+"-"+arg_list[2]+'^{2}'
    return expr_step

def Proportion(arg_list):
    """
        Proportion(a, b, c, d) -> a/b=c/d
        Proportion(a, b, c, d, e) -> (a/b)^e=c/d
    """
    if len(arg_list)<4:
        print("<Proportion> function has 4 augments at least!")
        raise Exception
    if len(arg_list)==4:
        expr_step = arg_list[0]+'*'+arg_list[3]+"-"+arg_list[1]+'*'+arg_list[2]
    else:
        expr_step = arg_list[0]+'*'+arg_list[3]+'^{1/'+arg_list[4]+"}-"+arg_list[1]+'*'+arg_list[2]+'^{1/'+arg_list[4]+"}"
    return expr_step

def Ratio(arg_list):
    """
        Ratio(a, b, c) -> a/b=c
        Ratio(a, b, c, d) -> (a/b)^c=d
    """
    if len(arg_list)<3 or len(arg_list)>4:
        print("<Power> function has 3 or 4 augments!")
        raise Exception
    if len(arg_list)==3:
        expr_step = arg_list[0]+' / '+arg_list[1]+'-'+arg_list[2]
    else:
        expr_step = '('+arg_list[0]+' / '+arg_list[1]+')^{'+arg_list[2]+"}"+"-"+arg_list[3]
    return expr_step

def Chord2_Ang(arg_list):
    """
        Chord2_Ang(a, b, c) -> a=(b+c)/2
    """
    if len(arg_list)!=3:
        print("<Chord2_Ang> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*2-'+arg_list[1]+'-'+arg_list[2]
    return expr_step

def TanSec_Ang(arg_list):
    """
        TanSec_Ang(a, b, c) -> a=(c-b)/2
    """
    if len(arg_list)!=3:
        print("<TanSec_Ang> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*2+'+arg_list[1]+'-'+arg_list[2]
    return expr_step

def Tria_BH_Area(arg_list):
    """
        Tria_BH_Area(a, b, c) -> a*b/2=c
    """
    if len(arg_list)!=3:
        print("<Tria_BH_Area> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[1]+'*0.5-'+arg_list[2]
    return expr_step

def Tria_SAS_Area(arg_list):
    """
        Tria_SAS_Area(a, b, c, d) -> a*c*sin(b)/2=d
    """
    if len(arg_list)!=4:
        print("<Tria_SAS_Area> function has 4 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[2]+'*0.5*\\sin{'+arg_list[1]+'/180*3.141593}-'+arg_list[3]
    return expr_step

def PRK_Perim(arg_list):
    """
        PRK_Perim(a, b, c) -> (a+b)*2=c 
    """
    if len(arg_list)!=3:
        print("<PRK_Perim> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*2+'+arg_list[1]+'*2-'+arg_list[2]
    return expr_step

def Para_Area(arg_list):
    """
        Para_Area(a, b, c) -> a*b=c
    """
    if len(arg_list)!=3:
        print("<Para_Area> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[1]+'-'+arg_list[2]
    return expr_step

def Rect_Area(arg_list):
    """
        Rect_Area(a, b, c) -> a*b=c
    """
    if len(arg_list)!=3:
        print("<Rect_Area> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[1]+'-'+arg_list[2]
    return expr_step

def Rhom_Area(arg_list):
    """
        Rhom_Area(a, b, c) -> a*b*2=c
    """
    if len(arg_list)!=3:
        print("<Phom_Area> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[1]+'*2-'+arg_list[2]
    return expr_step

def Kite_Area(arg_list):
    """
        Kite_Area(a, b, c) -> a*b/2=c
    """
    if len(arg_list)!=3:
        print("<Kite_Area> function has 3 augments!")
        raise Exception
    expr_step = arg_list[0]+'*'+arg_list[1]+'*0.5-'+arg_list[2]
    return expr_step

def Trap_Area(arg_list):
    """
        Trap_Area(a, b, c, d) -> (a+b)*c/2=d
    """
    if len(arg_list)!=4:
        print("<Trap_Area> function has 4 augments!")
        raise Exception
    expr_step = '0.5*('+arg_list[0]+'+'+arg_list[1]+')*'+arg_list[2]+'-'+arg_list[3]
    return expr_step

def Circle_R_Circum(arg_list):
    """
        Circle_R_Circum(a, b) -> 2*pi*a=b
        Circle_R_Circum(a, b, c) -> 2*pi*a*b/360=c
    """
    if len(arg_list)<2 or len(arg_list)>3:
        print("<Circle_Circum> function has 2 or 3 augments!")
        raise Exception
    if len(arg_list)==2:
        expr_step = '2*3.141593*'+arg_list[0]+'-'+arg_list[1]
    else:
        expr_step = '2*3.141593*'+arg_list[0]+'*'+arg_list[1]+'/360'+'-'+arg_list[2]
    return expr_step

def Circle_D_Circum(arg_list):
    """
        Circle_D_Circum(a, b) -> pi*a=b
        Circle_D_Circum(a, b, c) -> pi*a*b/360=c
    """
    if len(arg_list)<2 or len(arg_list)>3:
        print("<Circle_Circum> function has 2 or 3 augments!")
        raise Exception
    if len(arg_list)==2:
        expr_step = '3.141593*'+arg_list[0]+'-'+arg_list[1]
    else:
        expr_step = '3.141593*'+arg_list[0]+'*'+arg_list[1]+'/360'+'-'+arg_list[2]
    return expr_step

def Circle_R_Area(arg_list):
    """
        Circle_R_Area(a, b) -> pi*a^2=b
        Circle_R_Area(a, b, c) -> pi*a^2*b/360=c
    """
    if len(arg_list)<2 and len(arg_list)>3:
        print("<Circle_Area> function has 2 or 3 augments!")
        raise Exception
    if len(arg_list)==2:
        expr_step = '3.141593*'+arg_list[0]+'^{2}-'+arg_list[1]
    else:
        expr_step = '3.141593*'+arg_list[0]+'^{2}*'+arg_list[1]+'/360'+'-'+arg_list[2]
    return expr_step

def Circle_D_Area(arg_list):
    """
        Circle_D_Area(a, b) -> pi*(a/2)^2=b
        Circle_D_Area(a, b, c) -> pi*(a/2)^2*b/360=c
    """
    if len(arg_list)<2 and len(arg_list)>3:
        print("<Circle_Area> function has 2 or 3 augments!")
        raise Exception
    if len(arg_list)==2:
        expr_step = '0.25*3.141593*'+arg_list[0]+'^{2}-'+arg_list[1]
    else:
        expr_step = '0.25*3.141593*'+arg_list[0]+'^{2}*'+arg_list[1]+'/360'+'-'+arg_list[2]
    return expr_step

def ArcSeg_Area(arg_list):
    """
        ArcSeg_Area(a, b, c) -> pi*a^2*b/360 - a^2*sin(b)/2 = c
    """
    if len(arg_list)!=3:
        print("<ArcSeg_Area> function has 3 augments!")
        raise Exception
    expr_step = '3.141593*'+arg_list[0]+'^{2}*'+arg_list[1]+'/360-0.5*'+ \
                    arg_list[0]+'^{2}*\\sin{'+arg_list[1]+'/180*3.141593}-'+arg_list[2]
    return expr_step

def Ngon_Angsum(arg_list):
    """
        Ngon_Ang(a, b) -> (a-2)*180=b 
    """
    if len(arg_list)!=2:
        print("<Ngon_Ang> function has 2 augments!")
        raise Exception 
    expr_step = '('+arg_list[0]+'-2)*180-'+arg_list[1]
    return expr_step

def RNgon_B_Area(arg_list):
    """
        RNgon_B_Area(a, b, c) -> a*b^2/tan(180/a)/4=c
    """
    if len(arg_list)!=3:
        print("<RNgon_B_Area> function has 3 augments!")
        raise Exception 
    expr_step = arg_list[0]+'*'+arg_list[1]+'^{2}/4/\\tan{3.141593/'+arg_list[0]+'}-'+arg_list[2]
    return expr_step

def RNgon_L_Area(arg_list):
    """
        RNgon_L_Area(a, b, c) -> a*b^2*sin(360/a)/2=c
    """
    if len(arg_list)!=3:
        print("<RNgon_L_Area> function has 3 augments!")
        raise Exception 
    expr_step = arg_list[0]+'*'+arg_list[1]+'^{2}*0.5*\\sin{2*3.141593/'+arg_list[0]+'}-'+arg_list[2]
    return expr_step

def RNgon_H_Area(arg_list):
    """
        RNgon_H_Area(a, b, c) -> a*b^2*tan(180/a)=c
    """
    if len(arg_list)!=3:
        print("<RNgon_H_Area> function has 3 augments!")
        raise Exception 
    expr_step = arg_list[0]+'*'+arg_list[1]+'^{2}*\\tan{3.141593/'+arg_list[0]+'}-'+arg_list[2]
    return expr_step

def result_compute(num_all_list, exp_tokens):
    ANNO = Varible_Record()
    # Obtain the mapping between lowercase letters to intermediate variables V_i
    visit_list = [] # arguments denoted by lowercase letters
    for num in num_all_list:
        for item in spec_token_list:
            num = num.replace(item, "@"*len(item))
        for letter in num:
            if letter in low_case_list: visit_list.append(letter)
    for id, var in enumerate(num_all_list):
        ANNO.varible_dict["N"+str(id)] = var
        ANNO.mid_varible_dict["N"+str(id)] = var
    visit_list.sort() 
    no_visit_list = list(set(low_case_list)-set(spec_letter_list)-set(visit_list))
    no_visit_list.sort() # lowercase letters which have not used
    # mapping between letters to intermediate variables V_i
    midvar2letter = dict() 
    midletter2var = dict() 
    for id in range(V_NUM):
        midvar2letter['V'+str(id)] = no_visit_list[id]
        midletter2var[no_visit_list[id]] = 'V'+str(id)
    # step split
    step_list = []
    last_op_id = 0
    for id, token in enumerate(exp_tokens):
        if token in arith_op_list and id>0:
            step_list.append(exp_tokens[last_op_id:id])
            last_op_id = id
    step_list.append(exp_tokens[last_op_id:])
    # run step 
    for id, step in enumerate(step_list):
        operator = is_exist_operator(step[0], ANNO)
        if operator!='Get':
            operands = operand_update(step[1:], ANNO)
            expr_step = eval(operator)(operands)
            mid_var_solve(expr_step, ANNO, visit_list, midvar2letter)
            mid_var_update(ANNO, visit_list, midvar2letter, midletter2var, True)
            mid_var_update(ANNO, visit_list, midvar2letter, midletter2var, False)
        else: 
            Get(ANNO, step[1:])
    
    return ANNO.result

def normalize_exp(exp):
    # step split
    step_list = []
    last_op_id = 0
    for id, token in enumerate(exp):
        if token in arith_op_list and id>0:
            step_list.append(exp[last_op_id:id])
            last_op_id = id
    step_list.append(exp[last_op_id:])
    # normalize step
    new_exp = []
    for step in step_list:
        if step[0] in alterable_order_ops:
            if step[0] in ['Sum', 'Multiple']: 
                begin_id, end_id = 1, -1
                step[begin_id: end_id] = sorted(step[begin_id: end_id], key=lambda token:get_priority(token))
            if step[0] in ['Equal', 'Gougu', 'PRK_Perim', 'Rect_Area', 'Rhom_Area', 'Trap_Area']: 
                begin_id, end_id = 1, 3
                step[begin_id: end_id] = sorted(step[begin_id: end_id], key=lambda token:get_priority(token))
            if step[0] == 'Cos_Law': 
                begin_id, end_id = 2, 4
                step[begin_id: end_id] = sorted(step[begin_id: end_id], key=lambda token:get_priority(token))
            if step[0] in ['Sin_Law', 'Proportion']:
                if get_priority(step[1])>get_priority(step[3]) and len(step)==5:
                    step[1:3], step[3:5] = step[3:5], step[1:3]
            if step[0] in ['Tria_SAS_Area', 'Median']: 
                if get_priority(step[1])>get_priority(step[3]):
                    step[1], step[3] = step[3], step[1]
        new_exp += step

    return new_exp
