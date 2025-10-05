/** @type {import('next').NextConfig} */
const nextConfig = {
  // Don’t fail the build because of ESLint errors
  eslint: {
    ignoreDuringBuilds: true,
  },
};

module.exports = nextConfig;
