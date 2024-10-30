interface CardProps {
  title: string;
  description: string;
  icon: string;
}

const Card = ({ title, description, icon }: CardProps) => {
  return (
      <div className="rounded-2xl p-6 cursor-pointer transition-all hover:scale-105 h-full" 
           style={{ 
             background: 'linear-gradient(135deg, rgba(255,255,255,0.8), rgba(255,255,255,0.4))',
             backdropFilter: 'blur(10px)',
             boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
           }}>
        <img src={icon} alt={title} className="w-12 h-12 rounded-full mb-4"/>
        <h2 className="text-xl font-bold mb-2">{title}</h2>
        <p className="text-gray-600">{description}</p>
      </div>
  );
}

export default Card;
