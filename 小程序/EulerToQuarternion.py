from math import cos, sin, pi, atan2 ,asin
def EulerToQuaternion(x,y,z):

    #欧拉角(x, y, z)转换为四元数(q0, q1, q2, q3)
    #x , y , z 单位为角度,为固定坐标系
    # x, y, z  = 0, 0, 180
    x, y, z  = x*pi/180 ,y*pi/180 ,z*pi/180
    q0 ,q1 ,q2 ,q3 = 0 ,0 ,0 ,0
    q0 = cos(x/2)*cos(y/2)*cos(z/2) + sin(x/2)*sin(y/2)*sin(z/2)
    q1 = sin(x/2)*cos(y/2)*cos(z/2) - cos(x/2)*sin(y/2)*sin(z/2)
    q2 = cos(x/2)*sin(y/2)*cos(z/2) + sin(x/2)*cos(y/2)*sin(z/2)
    q3 = cos(x/2)*cos(y/2)*sin(z/2) - sin(x/2)*sin(y/2)*cos(z/2)
    print('欧拉角({0:f}, {1:f}, {2:f})转换为四元数(q0, q1, q2, q3)'.format(x*180/pi, y*180/pi, z*180/pi))
    print("q0 = {0:f}".format(q0))
    print("q1 = {0:f}".format(q1))
    print("q2 = {0:f}".format(q2))
    print("q3 = {0:f}".format(q3))
    print()

def QuaternionToEulerAngles(q0,q1,q2,q3):
    # 四元数q=(q0,q1,q2,q3)到欧拉角(x, y, z)
    # q0, q1, q2, q3 = 0.707, 0, 0, 0.707
    x, y, z = 0, 0, 0
    x = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)) * 180 / pi
    y = asin(2 * (q0 * q2 - q1 * q3)) * 180 / pi  # asin = arcsin
    z = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)) * 180 / pi
    print('四元数q=({0:f},{1:f},{2:f},{3:f})到欧拉角(x, y, z))'.format(q0, q1, q2, q3))
    print("x = {0:f}".format(x))
    print("y = {0:f}".format(y))
    print("z = {0:f}".format(z))
    print()
if __name__ == '__main__':
    EulerToQuaternion(30, 120, 30)
    QuaternionToEulerAngles(0.8, 0.035, 0.567,-0.117)