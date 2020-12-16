import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d as itp

class sphere:
    """A class which uses the Joukowski transform to transform a sphere in the 
    complex region to an ellipsoid in the real region.

    Parameters
    ----------
    input_vars : string or dict , optional
        Must be a .json file or python dictionary.

    Raises
    ------
    TypeError
        If the input_vars type is not a dictionary or the file path to 
        a .json file
    """
    def __init__(self,input_vars={}):

        # report
        print("Running 3D Conformal Mapping written by Ben Moulton\n")

        # get info or raise error
        self._get_input_vars(input_vars)

        # retrieve info
        self._retrieve_info()

        # create geometry
        self.create_geometry()
        
        # if stl desired, create
        if self.create_stl:
            self.export_stl()

        # if region desired, create
        if self.show_region:
            self.region()


    def _get_input_vars(self,input_vars):
        # get info or raise error

        # determine if the input_vars is a file or a dictionary
        input_vars_type = type(input_vars)

        # dictionary
        if input_vars_type == dict:
            self.input_dict = input_vars
        
        # json file
        elif input_vars_type == str and input_vars.split(".")[-1] == "json":
            self.input_dict = self._get_json(input_vars)

        # raise error
        else:
            raise TypeError("input_vars must be json file path, or " + \
                "dictionary, not {0}".format(input_vars_type))
    

    def _get_json(self,file_path):
        # import json file from file path
        json_string = open(file_path).read()

        # save to vals dictionary
        input_dict = json.loads(json_string)
        
        return input_dict


    def _retrieve_info(self):
        """A function which retrieves the information and stores it globally.
        """
        
        # store variables from file input dictionary
        
        # store geometry input values
        geometry = self.input_dict.get("geometry",{})
        self.sphere_radius = geometry.get("sphere_radius",2.0)
        self.vlambda = np.array(geometry.get("lambda",[0.0,0.5,0.5,0.5]))
        self.zeta0 = np.array(geometry.get("zeta_0",[0.0,0.0,0.0,0.0])) * \
            2. * self.sphere_radius
        self.n_theta = geometry.get("theta_nodes",10)
        self.n_phi = geometry.get("phi_nodes",10)
        
        # # store operating input values
        # operating = self.input_dict.get("operating",{})
        # self.freestream_velocity = operating.get("freestream_velocity",10.0)
        # angle_of_attack_deg = operating.get("angle_of_attack[deg]",15.0)
        # self.alpha = np.deg2rad(angle_of_attack_deg)
        # self.vortex_strength = operating.get("vortex_strength",0.0)

        # store run_commands input values
        run_commands = self.input_dict.get("run_commands",{})
        self.show_region = run_commands.get("show_region",False)
        self.create_stl = run_commands.get("create_stl",False)


    def geometry_sphere(self,theta,phi):
        """Determines the geometry at (a) given theta and phi location.

        Parameters
        ----------
        theta : float
            theta value(s) of interest.

        phi : float
            phi value(s) of interest.

        Returns
        -------
        point : quaternion
            Geometry quaternion coordinate in [m,x,y,z] format.
        """

        # defines coordinate values
        mu = self.zeta0[0]
        chi = self.sphere_radius * np.cos(theta) * np.sin(phi) + self.zeta0[1]
        psi = self.sphere_radius * np.sin(theta) * np.sin(phi) + self.zeta0[2]
        omega = self.sphere_radius * np.cos(phi) + self.zeta0[3]
        return np.array([ mu, chi, psi, omega])


    def geometry_ellipsoid(self,theta,phi):
        """Determines the geometry at (a) given theta and phi location.

        Parameters
        ----------
        theta : float
            theta value(s) of interest.

        phi : float
            phi value(s) of interest.

        Returns
        -------
        point : quaternion
            Geometry quaternion coordinate in [m,x,y,z] format.
        """

        # defines coordinate values
        mu = self.zeta0[0]
        chi = self.sphere_radius * np.cos(theta) * np.sin(phi) + self.zeta0[1]
        psi = self.sphere_radius * np.sin(theta) * np.sin(phi) + self.zeta0[2]
        omega = self.sphere_radius * np.cos(phi) + self.zeta0[3]

        mag = mu**2. + chi**2. + psi**2. + omega**2.
        Rn = self.vlambda
        # print((self.sphere_radius - self.vlambda*2.*self.sphere_radius)**2.)
        Np = 1. + Rn / mag
        Nm = 1. - Rn / mag

        return np.array([ mu*Np[0], chi*Nm[1], psi*Nm[2], omega*Nm[3]])


    def singularity_sphere(self,theta,phi):
        """Determines the singularity point at (a) given theta and phi location

        Parameters
        ----------
        theta : float
            theta value(s) of interest.

        phi : float
            phi value(s) of interest.

        Returns
        -------
        point : quaternion
            Geometry quaternion coordinate in [mu,chi,psi,omega] format.
        """

        # defines coordinate values
        m = self.zeta0[0]
        c = self.sphere_radius * np.cos(theta) * np.sin(phi) + self.zeta0[1]
        p = self.sphere_radius * np.sin(theta) * np.sin(phi) + self.zeta0[2]
        o = self.sphere_radius * np.cos(phi) + self.zeta0[3]

        # define lambda values
        lm = self.vlambda[0]
        lc = self.vlambda[1]
        lp = self.vlambda[2]
        lo = self.vlambda[3]

        # determine delta values
        mi = 1. / (m**2. + c**2. + p**2. + o**2.)
        dm = mi * (lm*m - lc*c - lp*p - lo*o)
        dc = mi * (lm*c + lc*m + lp*o - lo*p)
        dp = mi * (lm*p - lc*o + lp*m + lo*c)
        do = mi * (lm*o + lc*p - lp*c + lo*m)

        # determine nu values
        nm = ( ( dm + (dm**2. + dc**2. + dp**2. + do**2.)**0.5 ) / 2. )**0.5
        nc = dc / 2. / nm
        nps = dp / 2. / nm
        no = do / 2. / nm

        return -2. * np.array([nm, nc, nps, no])


    def create_geometry(self):
        """Create the requisite sphere and ellipsoid geometries.
        """


        # create theta and phi arrays
        theta = np.linspace(0.,2.*np.pi,self.n_theta)
        phi   = np.linspace(0.0,np.pi,self.n_phi)

        # initialize geometry 2d arrays
        sphere = np.zeros((self.n_theta,self.n_phi,4))
        ellipsoid = np.zeros((self.n_theta,self.n_phi,4))
        # oops = np.zeros((self.n_theta,self.n_phi,4))
        sin_sphere = np.zeros((self.n_theta,self.n_phi,4))

        for i in range(self.n_theta):
            for j in range(self.n_phi):
                sphere[i,j] = self.geometry_sphere(theta[i],phi[j])

                ellipsoid[i,j] = self.geometry_ellipsoid(theta[i],phi[j])

                # oops[i,j] = self.geometry_elloopsoid(theta[i],phi[j])

                sin_sphere[i,j] = self.singularity_sphere(theta[i],phi[j])
        
        # save these geometry arrays globally
        self.sphere = sphere
        self.ellipsoid = ellipsoid
        # self.oops = oops
        self.sin_sphere = sin_sphere
    

    def region(self):
        """Plot a sphere.
        """

        # report
        print("Plotting Region...\n")

        # plot a 3d plot of the wing
        fig = plt.figure()
        axe = fig.add_subplot(111,projection='3d')
        face = 'z'

        # initialize min and max vars
        xmin = 0; xmax = 0
        ymin = 0; ymax = 0
        zmin = 0; zmax = 0

        for i in range(self.n_theta):
            for j in range(self.n_phi):

                # determine maxs and mins of sphere
                if self.sphere[i,j,1] < xmin: xmin = self.sphere[i,j,1]
                if self.sphere[i,j,1] > xmax: xmax = self.sphere[i,j,1]
                if self.sphere[i,j,2] < ymin: ymin = self.sphere[i,j,2]
                if self.sphere[i,j,2] > ymax: ymax = self.sphere[i,j,2]
                if self.sphere[i,j,3] < zmin: zmin = self.sphere[i,j,3]
                if self.sphere[i,j,3] > zmax: zmax = self.sphere[i,j,3]

                # determine maxs and mins of ellipsoid
                if self.ellipsoid[i,j,1] < xmin: xmin = self.ellipsoid[i,j,1]
                if self.ellipsoid[i,j,1] > xmax: xmax = self.ellipsoid[i,j,1]
                if self.ellipsoid[i,j,2] < ymin: ymin = self.ellipsoid[i,j,2]
                if self.ellipsoid[i,j,2] > ymax: ymax = self.ellipsoid[i,j,2]
                if self.ellipsoid[i,j,3] < zmin: zmin = self.ellipsoid[i,j,3]
                if self.ellipsoid[i,j,3] > zmax: zmax = self.ellipsoid[i,j,3]

        # plot origin
        axe.scatter([0],[0],[0],zdir=face,c="k")

        # plot line to sphere center
        axe.plot([0,self.zeta0[1],self.zeta0[1],self.zeta0[1]],\
            [0,0,self.zeta0[2],self.zeta0[2]],\
                [0,0,0,self.zeta0[3]],zdir=face,c="g")

        # plot sphere center
        axe.scatter([self.zeta0[1]],[self.zeta0[2]],[self.zeta0[3]],\
            zdir=face,c="r")

        # plot the sphere
        for i in range(self.n_theta):
            axe.plot(self.sphere[i,:,1],self.sphere[i,:,2],\
                self.sphere[i,:,3],zdir=face,c="k")

        for j in range(self.n_phi):
            axe.plot(self.sphere[:,j,1],self.sphere[:,j,2],\
                self.sphere[:,j,3],zdir=face,c="k")
        
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                axe.scatter(self.sin_sphere[i,j,1],self.sin_sphere[i,j,2],\
                    self.sin_sphere[i,j,3],c="k")

        for i in range(self.n_theta):
            axe.plot(self.sin_sphere[i,:,1],self.sin_sphere[i,:,2],\
                self.sin_sphere[i,:,3],zdir=face,c="r")

        for j in range(self.n_phi):
            axe.plot(self.sin_sphere[:,j,1],self.sin_sphere[:,j,2],\
                self.sin_sphere[:,j,3],zdir=face,c="r")


        # plot the ellipsoid
        for i in range(self.n_theta):
            axe.plot(self.ellipsoid[i,:,1],self.ellipsoid[i,:,2],\
                self.ellipsoid[i,:,3],zdir=face,c="b")

        for j in range(self.n_phi):
            axe.plot(self.ellipsoid[:,j,1],self.ellipsoid[:,j,2],\
                self.ellipsoid[:,j,3],zdir=face,c="b")

        # solve for center
        xcent = (xmax + xmin) / 2.
        ycent = (ymax + ymin) / 2.
        zcent = (zmax + zmin) / 2.

        # solve for differences
        xdiff = np.abs(xmax - xmin)
        ydiff = np.abs(ymax - ymin)
        zdiff = np.abs(zmax - zmin)

        # solve for max difference
        max_diff = max([xdiff,ydiff,zdiff])

        # define limits
        x_lims = [xcent + 0.5*max_diff,xcent - 0.5*max_diff]
        y_lims = [ycent + 0.5*max_diff,ycent - 0.5*max_diff]
        z_lims = [zcent + 0.5*max_diff,zcent - 0.5*max_diff]

        # set limits
        axe.set_xlim3d(x_lims[1], x_lims[0])
        axe.set_ylim3d(y_lims[0], y_lims[1])
        axe.set_zlim3d(z_lims[1], z_lims[0])

        # set labels and finish plot
        axe.set_xlabel('X axis')
        axe.set_ylabel('Y axis')
        axe.set_zlabel('Z axis')
        # axe.view_init(200,-60)
        axe.view_init(20,-120)
        plt.show()


    def make_stl(self,object):
        """Method which creates an stl object from an object of the 2d array
        shape as created in the create geometry function.

        Parameters
        ----------
        object : 2d array
            A 2d array as created in the create_geometry method.
        
        Returns
        -------
        normals : array
            An array of the face normals.
        
        vertices : array
            An array of vertices for the object.

        faces : array
            A 2D array of indices referencing the vertices to create faces.
        """

        # create vertices array
        vertices = np.zeros((self.n_vertices,3))

        # initialize vertices counter
        k = 1

        # add in first and final points
        vertices[0] = object[0,0,1:]
        vertices[-1] = object[0,-1,1:]

        # add vertices
        for i in range(self.n_theta - 1):
            for j in range(1,self.n_phi-1):
                vertices[k] = object[i,j,1:]
                k += 1
    
        # create faces array
        faces = np.zeros((self.n_faces,3))

        # initialize faces counter
        k = 0
        
        # add faces
        for i in range(self.n_theta - 2):

            # add top triangle
            faces[k] = np.array([0,int((i+1) * (self.n_phi - 2)) + 1,\
                int(i * (self.n_phi - 2)) + 1])
            k += 1

            # add in between squares
            for j in range(1,self.n_phi-2):
                # add first triangle
                faces[k] = np.array([int(i * (self.n_phi - 2)) + j,\
                    int((i+1) * (self.n_phi - 2)) + j,\
                        int((i+1) * (self.n_phi - 2)) + (j+1)])
                k += 1
                
                # add second triangle
                faces[k] = np.array([int((i+1) * (self.n_phi - 2)) + (j+1),\
                        int(i * (self.n_phi - 2)) + (j+1),\
                            int(i * (self.n_phi - 2)) + j])
                k += 1
                
            # add bottom triangle
            faces[k] = np.array([self.n_vertices -1,\
                int((i+1) * (self.n_phi - 2)),\
                    int((i+2) * (self.n_phi - 2))])
            k += 1

        ## final segment; initialize i counter
        i = self.n_theta - 2

        # add top triangle
        faces[k] = np.array([0,1,int(i * (self.n_phi - 2)) + 1])
        k += 1

        # add in between squares
        for j in range(1,self.n_phi-2):
            # add first triangle
            faces[k] = np.array([int(i * (self.n_phi - 2)) + j,j,j+1])
            k += 1
            
            # add second triangle
            faces[k] = np.array([j+1,int(i * (self.n_phi - 2)) + (j+1),\
                        int(i * (self.n_phi - 2)) + j])
            k += 1
            
        # add bottom triangle
        faces[k] = np.array([self.n_vertices -1,\
            int((i+1) * (self.n_phi - 2)),\
                int((self.n_phi - 2))])
        k += 1

        # create normals array
        normals = np.zeros((self.n_faces,3))

        # run through and create each normal
        for i in range(self.n_faces):

            # create point vectors
            vec2 = vertices[int(faces[i,1])] - vertices[int(faces[i,0])]
            vec1 = vertices[int(faces[i,2])] - vertices[int(faces[i,0])]

            # cross vectors
            normal = np.cross(vec1,vec2)
            first = normal

            # make unit vector
            normals[i] = normal / np.linalg.norm(normal)


        return normals, vertices, faces


    def write_stl(self,normals,vertices,faces,file_name):
        """Method which writes the stl file for export.

        Parameters:
        normals : array
            An array of the face normals.
        
        vertices : array
            An array of vertices for the object.

        faces : array
            A 2D array of indices referencing the vertices to create faces.
        
        file_name : string
            The file name to create. ex. 'sphere.stl'
        """

        # write to file
        with open(file_name,"w") as f:

            # write solid intro
            f.write("solid " + file_name[:-4] + "\n")

            # write each triangle
            for i in range(faces.shape[0]):

                # write facet intro
                f.write("\tfacet normal")

                # write facet normal
                for j in range(3):
                    val = np.format_float_scientific(normals[i,j],\
                        unique=False,trim="k",pad_left=3,precision=6,\
                            exp_digits=3)
                    f.write(val)
                f.write("\n")

                # write loop intro
                f.write("\t\touter loop\n")

                # write each vertex
                for j in range(3):

                    f.write("\t\t\tvertex\t")

                    # write each coordinate
                    for k in range(3):
                        val = np.format_float_scientific(\
                            vertices[int(faces[i,j]),k],unique=False,\
                                trim="k",pad_left=3,precision=6,exp_digits=3)
                        f.write(val)
                    f.write("\n")

                # write loop outro
                f.write("\t\tendloop\n")

                # write facet intro
                f.write("\tendfacet \n")

            # write solid outro
            f.write("endsolid " + file_name[:-4] + "\n")

            # close file
            f.close()


    def export_stl(self):
        """Method which exports the stl  of the sphere and ellipsoid
        """

        # report
        print("Creating and exporting sphere object as sphere.stl...\n")

        # calculate number of vertices
        self.n_vertices = self.n_phi + (self.n_phi - 2) * (self.n_theta - 2)

        # calculate number of faces
        self.n_faces = (2 * (self.n_phi - 2) ) * (self.n_theta - 1)

        # create sphere object
        sphere_normals,sphere_vertices,sphere_faces= self.make_stl(self.sphere)

        # write stl file
        self.write_stl(sphere_normals, sphere_vertices, sphere_faces, \
            "sphere.stl")

        # create ellipsoid object
        ellipsoid_normals, ellipsoid_vertices, ellipsoid_faces = \
            self.make_stl(self.ellipsoid)

        # write stl file
        self.write_stl(ellipsoid_normals, ellipsoid_vertices, \
            ellipsoid_faces,"ellipsoid.stl") 
