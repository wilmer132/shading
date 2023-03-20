//
// Parameters that control fragment shader behavior. Different materials
// will set these flags to true/false for different looks
//

uniform bool useTextureMapping;     // true if basic texture mapping (diffuse) should be used
uniform bool useNormalMapping;      // true if normal mapping should be used
uniform bool useEnvironmentMapping; // true if environment mapping should be used
uniform bool useMirrorBRDF;         // true if mirror brdf should be used (default: phong)

//
// texture maps
//

uniform sampler2D diffuseTextureSampler;

// CS248 Part 3: Normal Mapping
uniform sampler2D normalMappingSampler;

// CS248 Part 4: Environment Mapping
uniform sampler2D environmentMappingSampler;

//
// lighting environment definition. Scenes may contain directional
// and point light sources, as well as an environment map
//

#define MAX_NUM_LIGHTS 10
uniform int  num_directional_lights;
uniform vec3 directional_light_vectors[MAX_NUM_LIGHTS];

uniform int  num_point_lights;
uniform vec3 point_light_positions[MAX_NUM_LIGHTS];

uniform int   num_spot_lights;
uniform vec3  spot_light_positions[MAX_NUM_LIGHTS];
uniform vec3  spot_light_directions[MAX_NUM_LIGHTS];
uniform vec3  spot_light_intensities[MAX_NUM_LIGHTS];
uniform float spot_light_angles[MAX_NUM_LIGHTS];


//
// material-specific uniforms
//

// parameters to Phong BRDF
uniform float spec_exp;

// values that are varying per fragment (computed by the vertex shader)

in vec3 position;     // surface position
in vec3 normal;
in vec2 texcoord;     // surface texcoord (uv)
in vec3 dir2camera;   // vector from surface point to camera
in mat3 tan2world;    // tangent space to world space transform
in vec3 vertex_diffuse_color; // surface color

out vec4 fragColor;

#define PI 3.14159265358979323846


//
// Simple diffuse brdf
//
// L -- direction to light
// N -- surface normal at point being shaded
//
vec3 Diffuse_BRDF(vec3 L, vec3 N, vec3 diffuseColor) {
    return diffuseColor * max(dot(N, L), 0.);
}

//
// Phong_BRDF --
//
// Evaluate phong reflectance model according to the given parameters
// L -- direction to light
// V -- direction to camera (view direction)
// N -- surface normal at point being shaded
//
vec3 Phong_BRDF(vec3 L, vec3 V, vec3 N, vec3 diffuse_color, vec3 specular_color, float specular_exponent)
{
    // CS248 Part 2: Phong Reflectance
    // Implement diffuse and specular terms of the Phong
    // reflectance model here.

    // Normalize provided vectors
    vec3 L_norm = normalize(L);
    vec3 V_norm = normalize(V);
    vec3 N_norm = normalize(N);
    
    // Direction vector
    vec3 R = (2 * dot(L_norm, N_norm) * N_norm) - L_norm;

    // Components for diffuse and specular
    vec3 diffuse = diffuse_color * max (0, dot(L_norm, N_norm));
    vec3 specular = specular_color * pow(max (0, dot(R, V_norm)), specular_exponent);

    vec3 result = diffuse + specular;

    return result;
}

//
// SampleEnvironmentMap -- returns incoming radiance from specified direction
//
// D -- world space direction (outward from scene) from which to sample radiance
// 
vec3 SampleEnvironmentMap(vec3 D)
{    
    // CS248 Part 4: Environment Mapping
    // sample environment map in direction D.  This requires
    // converting D into spherical coordinates where Y is the polar direction
    // (warning: in our scene, theta is angle with Y axis, which differs from
    // typical convention in physics)
    //
    // Tips:
    //
    // (1) See GLSL documentation of acos(x) and atan(x, y)
    //
    // (2) atan() returns an angle in the range -PI to PI, so you'll have to
    //     convert negative values to the range 0 - 2PI
    //
    // (3) How do you convert theta and phi to normalized texture
    //     coordinates in the domain [0,1]^2?

    vec3 D_norm = normalize(D);

    // Right handed coordinates from Cartesian to Sphere
    // X: Towards Us -> Towards Right
    // Y: Towards Right -> Up
    // Z: Up -> Towards Us
    float theta = acos(D_norm.y);           // Determine vertical position
    float phi = atan(D_norm.x, D_norm.z);   // Determines horiozontal position

    // Change all values to be between 0 and 1
    float theta_norm = theta / PI;
    float phi_norm = ((2 * PI - phi) / (2 * PI));

    // Use spherical normed coordinate to pull from map
    vec2 envcoord = vec2(phi_norm, theta_norm);
    vec3 texture_value = texture(environmentMappingSampler, envcoord).rgb;
    
    return texture_value;  
}

//
// Fragment shader main entry point
//
void main(void)
{

    //////////////////////////////////////////////////////////////////////////
	// Pattern generation. Compute parameters to BRDF 
    //////////////////////////////////////////////////////////////////////////
    
	vec3 diffuseColor = vec3(1.0, 1.0, 1.0);
    vec3 specularColor = vec3(1.0, 1.0, 1.0);
    float specularExponent = spec_exp;

    if (useTextureMapping) {
        diffuseColor = texture(diffuseTextureSampler, texcoord).rgb;
    } else {
        diffuseColor = vertex_diffuse_color;
    }

    // perform normal map lookup if required
    vec3 N = vec3(0);
    if (useNormalMapping) {
       // CS248 Part 3: Normal Mapping:
       // use tan2World in the normal map to compute the
       // world space normal baaed on the normal map.

       // Note that values from the texture should be scaled by 2 and biased
       // by negative -1 to covert positive values from the texture fetch, which
       // lie in the range (0-1), to the range (-1,1).
       //
       // In other words:   tangent_space_normal = texture_value * 2.0 - 1.0;

       vec3 texture_value = texture(normalMappingSampler, texcoord).rgb;
       vec3 tangent_space_normal = texture_value * 2.0 - 1.0;
       vec3 world_space_normal = tan2world * tangent_space_normal;

       N = normalize(world_space_normal);

    } else {
       N = normalize(normal);
    }

    vec3 V = normalize(dir2camera);
    vec3 Lo = vec3(0.1 * diffuseColor);   // this is ambient

    /////////////////////////////////////////////////////////////////////////
    // Phase 2: Evaluate lighting and surface BRDF 
    /////////////////////////////////////////////////////////////////////////

    if (useMirrorBRDF) {
        //
        // CS248 Environment Mapping:
        // compute perfect mirror reflection direction here.
        // You'll also need to implement environment map sampling in SampleEnvironmentMap()
        //
        // Find angle going away from object to camera
        vec3 w_i = normalize(dir2camera); // Make sure it is normal

        // Use equation from lecture
        // Find angle going away from object towards light
        vec3 R = -w_i + (2 * dot(w_i, N) * N);

        // Get color / reflectance
        vec3 envColor = SampleEnvironmentMap(R);
        
        // this is a perfect mirror material, so we'll just return the light incident
        // from the reflection direction
        fragColor = vec4(envColor, 1);
        return;
    }

	// for simplicity, assume all lights (other than spot lights) have unit magnitude
	float light_magnitude = 1.0;

	// for all directional lights
	for (int i = 0; i < num_directional_lights; ++i) {
	    vec3 L = normalize(-directional_light_vectors[i]);
		vec3 brdf_color = Phong_BRDF(L, V, N, diffuseColor, specularColor, specularExponent);
	    Lo += light_magnitude * brdf_color;
    }

    // for all point lights
    for (int i = 0; i < num_point_lights; ++i) {
		vec3 light_vector = point_light_positions[i] - position;
        vec3 L = normalize(light_vector);
        float distance = length(light_vector);
        vec3 brdf_color = Phong_BRDF(L, V, N, diffuseColor, specularColor, specularExponent);
        float falloff = 1.0 / (0.01 + distance * distance);
        Lo += light_magnitude * falloff * brdf_color;
    }

    // for all spot lights
	for (int i = 0; i < num_spot_lights; ++i) {
    
        vec3 intensity = spot_light_intensities[i];   // intensity of light: this is intensity in RGB
        vec3 light_pos = spot_light_positions[i];     // location of spotlight
        float cone_angle = spot_light_angles[i];      // spotlight falls off to zero in directions whose
                                                      // angle from the light direction is grester than
                                                      // cone angle. Caution: this value is in units of degrees!

        vec3 dir_to_surface = position - light_pos;
        float angle = acos(dot(normalize(dir_to_surface), spot_light_directions[i])) * 180.0 / PI;

        // TODO CS248 Part 5.1: Spotlight Attenuation: compute the attenuation of the spotlight due to two factors:
        // (1) distance from the spot light (D^2 falloff)
        // (2) attentuation due to being outside the spotlight's cone 
        //
        // Here is a description of what to compute:
        //
        // 1. Modulate intensity by a factor of 1/D^2, where D is the distance from the
        //    spotlight to the current surface point.  For robustness, it's common to use 1/(1 + D^2)
        //    to never multiply by a value greather than 1.

        float int_fact = 1.0 / (1.0 + dot(dir_to_surface, dir_to_surface));
        intensity *= int_fact;

        // 2. Modulate the resulting intensity based on whether the surface point is in the cone of
        //    illumination.  To achieve a smooth falloff, consider the following rules
        //    
        //    -- Intensity should be zero if angle between the spotlight direction and the vector from
        //       the light position to the surface point is greater than (1.0 + SMOOTHING) * cone_angle
        //
        //    -- Intensity should not be further attentuated if the angle is less than (1.0 - SMOOTHING) * cone_angle
        //
        //    -- For all other angles between these extremes, interpolate linearly from unattenuated
        //       to zero intensity. 
        //
        //    -- The reference solution uses SMOOTHING = 0.1, so 20% of the spotlight region is the smoothly
        //       facing out area.  Smaller values of SMOOTHING will create hard spotlights.
        float SMOOTHING = 0.1;
        float SHADOW_EDGE_UPPER = (1.0 + SMOOTHING) * cone_angle;
        float SHADOW_EDGE_LOWER = (1.0 - SMOOTHING) * cone_angle;
        
        if (angle > SHADOW_EDGE_UPPER) {
          intensity *= 0;
        } else if (angle >= SHADOW_EDGE_LOWER && angle <= SHADOW_EDGE_UPPER) {
          float diff = SHADOW_EDGE_UPPER - SHADOW_EDGE_LOWER;
          float lerp_int = (angle - SHADOW_EDGE_UPPER) / (-diff);
          intensity *= lerp_int;
        }

        // Render Shadows for all spot lights
        // TODO CS248 Part 5.2: Shadow Mapping: comute shadowing for spotlight i here 


	    vec3 L = normalize(-spot_light_directions[i]);
		vec3 brdf_color = Phong_BRDF(L, V, N, diffuseColor, specularColor, specularExponent);

	    Lo += intensity * brdf_color;
    }

    fragColor = vec4(Lo, 1);
}



