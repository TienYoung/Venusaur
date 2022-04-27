#include "Camera.h"

#include "Common.h"

const glm::vec3& Camera::m_defaultForward = glm::vec3(0.0f, 0.0f, -1.0f);

Camera::Camera()
{
	UpdateUVW();
}

Camera::Camera(const glm::vec3& origin, float vfov, float aspect, float aperture, float focal_length)
	: m_position(origin), m_vfov(vfov), m_aspect(aspect), m_aperture(aperture), m_focalLength(focal_length)
{
	UpdateUVW();
}

void Camera::UVWFrame(glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
	UpdateUVW();
	U = m_u;
	V = m_v;
	W = m_w;
}

void Camera::UpdateUVW()
{
	m_w = m_forward * m_focalLength;
	m_u = normalize(cross(m_w, WORLDUP));
	m_v = normalize(cross(m_u, m_w));

	float theta = glm::radians(m_vfov);
	float h = glm::tan(theta * 0.5f);
	float viewportHeight = 2.0f * h;
	float viewportWidth = m_aspect * viewportHeight;

	m_u *= m_focalLength * viewportWidth;
	m_v *= m_focalLength * viewportHeight;
}

